use crate::cache::Cache;
use crate::nn_loader::*;
use crate::rotary::RotaryEmbedding;
use crate::tensor_loader::SafeTensorLoader;
use anyhow::Result;
use dfdx::dtypes::{f16, AMP};
use dfdx::prelude::*;
use num_traits::Float;
use std::fmt::Debug;

pub trait Dtype: dfdx::prelude::Dtype + num_traits::Float {}
impl Dtype for f32 {}
impl Dtype for f16 {}
impl Dtype for AMP<f16> {}

#[allow(clippy::upper_case_acronyms)]
struct MHA<E: Dtype, P: Params, D: Device<E>> {
    wq: Linear<P::Hidden, P::Hidden, E, D>,
    wk: Linear<P::Hidden, P::Hidden, E, D>,
    wv: Linear<P::Hidden, P::Hidden, E, D>,
    out_proj: Linear<P::Hidden, P::Hidden, E, D>,
    p: P,
}
#[allow(clippy::type_complexity)]
impl<E: Dtype, P: Params, D: Device<E>> MHA<E, P, D>
where
    D: Device<f32> + ToDtypeKernel<E, f32> + ToDtypeKernel<f32, E>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::Hidden), E, D>,
        layer: usize,
        pos: usize,
        pos_scale: usize,
        pos_enc: &RotaryEmbedding<P::RoeDim, E, D>,
        cache: &mut Option<&mut Cache<P::Heads, P::HeadDim, P::Layers, E, D>>,
        mask: &mut Option<Tensor<(P::Heads, usize, usize), f32, D>>,
    ) -> Result<Tensor<(Seq, P::Hidden), E, D>, Error> {
        let dev = x.dev().clone();
        let (seq, hidden) = *x.shape();
        let q = self.wq.try_forward(x.clone())?;
        let k = self.wk.try_forward(x.clone())?;
        let v = self.wv.try_forward(x)?;

        let qkvs = (seq.size(), self.p.heads(), self.p.head_dim());
        let q = q.reshape_like(&qkvs);
        let k = k.reshape_like(&qkvs);
        let mut v = v.reshape_like(&qkvs);

        let q = pos_enc.try_forward(q, pos, pos_scale)?;
        let mut k = pos_enc.try_forward(k, pos, pos_scale)?;

        if let Some(cache) = cache.as_mut() {
            (k, v) = cache.append(layer, k, v);
        }

        let (kv_seq, _headers, _header_dim) = k.shape().concrete().into();

        //(seq, header, header_dim) -> (headers, seq, header_dim)
        let q = q.permute::<_, Axes3<1, 0, 2>>().to_dtype::<f32>();
        let k = k.permute::<_, Axes3<1, 0, 2>>().to_dtype::<f32>();
        let v = v.permute::<_, Axes3<1, 0, 2>>().to_dtype::<f32>();

        let scale = (self.p.head_dim().size() as f64).sqrt().recip();
        let mut att = q.matmul(k.permute::<_, Axes3<0, 2, 1>>()) * scale;

        let mask = {
            let _mask = match mask {
                Some(mask) => mask.clone(),
                None => {
                    let attn_seq = kv_seq;
                    let mask = dev.upper_tri_like(&(attn_seq, attn_seq), f32::neg_infinity(), 1);
                    let sub_mask_sel = ((attn_seq - seq.size())..attn_seq).collect();
                    let sub_mask_sel = dev.tensor_from_vec(sub_mask_sel, (seq.size(),));
                    mask.gather(sub_mask_sel).broadcast_like(&att).realize()
                }
            };
            *mask = Some(_mask.clone());
            _mask
        };

        att = mask.clone() + att;

        let att = att.softmax::<Axis<2>>();

        let v: Tensor<(Seq, P::Hidden), _, _> = att
            .matmul(v)
            .permute::<_, Axes3<1, 0, 2>>()
            //.contiguous()
            .try_reshape_like(&(seq, hidden))?
            .realize();

        let out = self.out_proj.try_forward(v.to_dtype())?;

        Ok(out)
    }
}

#[allow(clippy::upper_case_acronyms)]
struct MLP<E: Dtype, P: Params, D: Device<E>> {
    fc1: Linear<P::Hidden, P::MlpDim, E, D>,
    fc2: Linear<P::MlpDim, P::Hidden, E, D>,
}
impl<E: Dtype, P: Params, D: Device<E>> MLP<E, P, D> {
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::Hidden), E, D>,
    ) -> Result<Tensor<(Seq, P::Hidden), E, D>> {
        let x = self.fc1.try_forward(x)?;
        let x = x.fast_gelu();
        let x = self.fc2.try_forward(x)?;
        Ok(x)
    }
}
struct Block<E: Dtype, P: Params, D: Device<E>>
where
    D: Device<f32>,
{
    ln: LayerNorm1D<P::Hidden, f32, D>,
    mha: MHA<E, P, D>,
    mlp: MLP<E, P, D>,
}

#[allow(clippy::type_complexity)]
impl<E: Dtype, P: Params, D: Device<E>> Block<E, P, D>
where
    D: Device<f32> + ToDtypeKernel<E, f32> + ToDtypeKernel<f32, E>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::Hidden), E, D>,
        layer: usize,
        pos: usize,
        pos_scale: usize,
        pos_enc: &RotaryEmbedding<P::RoeDim, E, D>,
        cache: &mut Option<&mut Cache<P::Heads, P::HeadDim, P::Layers, E, D>>,
        mask: &mut Option<Tensor<(P::Heads, usize, usize), f32, D>>,
    ) -> Result<Tensor<(Seq, P::Hidden), E, D>> {
        let residual = x.clone();
        let x = self.ln.try_forward(x.to_dtype())?.to_dtype();
        let attn_out =
            self.mha
                .try_forward(x.clone(), layer, pos, pos_scale, pos_enc, cache, mask)?;
        let mlp_out = self.mlp.try_forward(x)?;
        Ok(residual + attn_out + mlp_out)
    }
}

pub struct PhiLM<E: Dtype, P: Params, D: Device<E>>
where
    D: Device<f32>,
{
    embedding: Embedding<P::Vocab, P::Hidden, E, D>,
    blocks: Vec<Block<E, P, D>>,
    lm_ln: LayerNorm1D<P::Hidden, f32, D>,
    lm_linear: Linear<P::Hidden, P::Vocab, E, D>,
    pos_enc: RotaryEmbedding<P::RoeDim, E, D>,
    p: P,
}

impl<E: Dtype, P: Params, D: Device<E>> PhiLM<E, P, D>
where
    D: Device<f32>,
{
    pub fn load_model(p: P, dev: &D, loader: &SafeTensorLoader) -> Result<Self>
    where
        D: Device<f16> + ToDtypeKernel<f16, E>,
    {
        let loader = loader.sub("model");
        let embedding = load_emedding(dev, &loader.sub("embed_tokens"))?;

        let mut blocks = Vec::new();

        for i in 0..p.layers().size() {
            let loader = loader.sub(format!("layers.{i}"));
            let ln = load_layernorm(dev, &loader.sub("input_layernorm"))?;
            let wq = load_linear(dev, &loader.sub("self_attn.q_proj"))?;
            let wk = load_linear(dev, &loader.sub("self_attn.k_proj"))?;
            let wv = load_linear(dev, &loader.sub("self_attn.v_proj"))?;
            let out_proj = load_linear(dev, &loader.sub("self_attn.dense"))?;
            let fc1 = load_linear(dev, &loader.sub("mlp.fc1"))?;
            let fc2 = load_linear(dev, &loader.sub("mlp.fc2"))?;

            blocks.push(Block {
                ln,
                mha: MHA {
                    wq,
                    wk,
                    wv,
                    out_proj,
                    p,
                },
                mlp: MLP { fc1, fc2 },
            });
        }

        let lm_ln = load_layernorm(dev, &loader.sub("final_layernorm"))?;
        let lm_linear = load_linear(dev, &loader.root().sub("lm_head"))?;

        let pos_enc = RotaryEmbedding::new(dev, p.roe_dim(), P::MAX_SEQ_LEN, P::ROE_BASE);

        Ok(Self {
            embedding,
            blocks,
            lm_ln,
            lm_linear,
            pos_enc,
            p,
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq,), usize, D>,
        pos: usize,
        pos_scale: usize,
        cache: &mut Option<&mut Cache<P::Heads, P::HeadDim, P::Layers, E, D>>,
    ) -> Result<Tensor<(Seq, P::Vocab), E, D>>
    where
        D: Device<f32> + ToDtypeKernel<E, f32> + ToDtypeKernel<f32, E>,
    {
        let mut x = self.embedding.try_forward(x)?;
        let mut mask = None;
        for (i, b) in self.blocks.iter().enumerate() {
            x = b.try_forward(x, i, pos, pos_scale, &self.pos_enc, cache, &mut mask)?;
        }
        let x = self.lm_ln.try_forward(x.to_dtype())?.to_dtype();
        let x = self.lm_linear.try_forward(x)?;
        Ok(x)
    }

    pub fn params(&self) -> &P {
        &self.p
    }
}

pub trait Params: Debug + Clone + Copy {
    type Vocab: ConstDim;
    type Hidden: ConstDim;
    type MlpDim: ConstDim;
    type Heads: ConstDim;
    type HeadDim: ConstDim;
    type QkvDim: ConstDim;
    type Layers: ConstDim;
    type RoeDim: ConstDim;
    const MAX_SEQ_LEN: usize;
    const ROE_BASE: i64;

    fn vocab(&self) -> Self::Vocab;
    fn hidden(&self) -> Self::Hidden;
    fn mlp_dim(&self) -> Self::MlpDim;
    fn heads(&self) -> Self::Heads;
    fn head_dim(&self) -> Self::HeadDim;
    fn qkv_dim(&self) -> Self::QkvDim;
    fn layers(&self) -> Self::Layers;
    fn roe_dim(&self) -> Self::RoeDim;
}
