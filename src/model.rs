use crate::cache::Cache;
use crate::nn_loader::*;
use crate::rmsnorm::RmsNorm;
use crate::rotary::RotaryEmbedding;
use crate::tensor_loader::SafeTensorLoader;
use anyhow::Result;
use dfdx::dtypes::{f16, AMP};
use dfdx::prelude::*;
use std::fmt::Debug;

pub trait Dtype: dfdx::prelude::Dtype + num_traits::Float {}
impl Dtype for f32 {}
impl Dtype for f16 {}
impl Dtype for AMP<f16> {}

#[allow(clippy::upper_case_acronyms)]
struct MHA<E: Dtype, P: Params, D: Device<E>> {
    k_proj: Linear<P::Hidden, P::KvDim, E, D>,
    q_proj: Linear<P::Hidden, P::Hidden, E, D>,
    v_proj: Linear<P::Hidden, P::KvDim, E, D>,
    o_proj: Linear<P::Hidden, P::Hidden, E, D>,
    p: P,
}
#[allow(clippy::type_complexity)]
impl<E: Dtype, P: Params, D: Device<E>> MHA<E, P, D> {
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::Hidden), E, D>,
        layer: usize,
        pos: usize,
        pos_scale: usize,
        pos_enc: &RotaryEmbedding<P::HeadDim, E, D>,
        mut cache: Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
        mask: Option<Tensor<(P::Heads, usize, usize), E, D>>,
    ) -> Result<
        (
            Tensor<(Seq, P::Hidden), E, D>,
            Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
            Option<Tensor<(P::Heads, usize, usize), E, D>>,
        ),
        Error,
    > {
        let dev = x.dev().clone();
        let (seq, hidden) = *x.shape();
        let q = self.q_proj.try_forward(x.clone())?;
        let k = self.k_proj.try_forward(x.clone())?;
        let v = self.v_proj.try_forward(x)?;

        let qs = (seq.size(), self.p.heads(), self.p.head_dim());
        let q = pos_enc.try_forward(q.reshape_like(&qs), pos, pos_scale)?;

        let kvs = (seq.size(), self.p.kv_heads(), self.p.head_dim());
        let mut k = pos_enc.try_forward(k.reshape_like(&kvs), pos, pos_scale)?;

        let mut v = v.reshape_like(&kvs);

        if let Some(cache) = cache.as_mut() {
            (k, v) = cache.append(layer, k, v);
        }

        let repeate = self.p.heads().size() / self.p.kv_heads().size();
        let (kv_seq, _headers, _header_dim) = k.shape().concrete().into();
        let kvs2 = (kv_seq, self.p.kv_heads(), repeate, self.p.head_dim());
        let kvs3 = (kv_seq, self.p.heads(), self.p.head_dim());

        //(seq, header, header_dim) -> (headers, seq, header_dim)
        let q = q.permute::<_, Axes3<1, 0, 2>>();
        let k = k
            .broadcast_like(&kvs2)
            .try_reshape_like(&kvs3)?
            .permute::<_, Axes3<1, 0, 2>>();
        let v = v
            .broadcast_like(&kvs2)
            .try_reshape_like(&kvs3)?
            .permute::<_, Axes3<1, 0, 2>>();

        let scale = (self.p.head_dim().size() as f64).sqrt().recip();
        let mut att = q.matmul(k.permute::<_, Axes3<0, 2, 1>>()) * scale;

        let mask = match mask {
            Some(mask) => mask,
            None => {
                let attn_seq = kv_seq;
                let mask = dev.upper_tri_like(&(attn_seq, attn_seq), E::neg_infinity(), 1);
                let sub_mask_sel = ((attn_seq - seq.size())..attn_seq).collect();
                let sub_mask_sel = dev.tensor_from_vec(sub_mask_sel, (seq.size(),));
                mask.gather(sub_mask_sel).broadcast_like(&att).realize()
            }
        };
        att = mask.clone() + att;

        let att = att.softmax::<Axis<2>>();

        let v: Tensor<(Seq, P::Hidden), _, _> = att
            .matmul(v)
            .permute::<_, Axes3<1, 0, 2>>()
            .try_reshape_like(&(seq, hidden))?
            .realize();

        let out = self.o_proj.try_forward(v)?;

        Ok((out, cache, Some(mask)))
    }
}

#[allow(clippy::upper_case_acronyms)]
struct MLP<E: Dtype, P: Params, D: Device<E>> {
    fc1: Linear<P::Hidden, P::MlpDim, E, D>,
    fc2: Linear<P::Hidden, P::MlpDim, E, D>,
    proj: Linear<P::MlpDim, P::Hidden, E, D>,
}
impl<E: Dtype, P: Params, D: Device<E>> MLP<E, P, D> {
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::Hidden), E, D>,
    ) -> Result<Tensor<(Seq, P::Hidden), E, D>> {
        let y1 = self.fc1.try_forward(x.clone())?;
        let y1 = y1.clone().sigmoid() * y1;
        let y2 = self.fc2.try_forward(x)?;
        let y = self.proj.try_forward(y1 * y2)?;
        Ok(y)
    }
}
struct Block<E: Dtype, P: Params, D: Device<E>>
where
    D: Device<f32>,
{
    rms_1: RmsNorm<P::Hidden, f32, D>,
    rms_2: RmsNorm<P::Hidden, f32, D>,
    mha: MHA<E, P, D>,
    mlp: MLP<E, P, D>,
}

#[allow(clippy::type_complexity)]
impl<E: Dtype, P: Params, D: Device<E>> Block<E, P, D>
where
    D: Device<f32>,
{
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::Hidden), E, D>,
        layer: usize,
        pos: usize,
        pos_scale: usize,
        pos_enc: &RotaryEmbedding<P::HeadDim, E, D>,
        cache: Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
        mask: Option<Tensor<(P::Heads, usize, usize), E, D>>,
    ) -> Result<(
        Tensor<(Seq, P::Hidden), E, D>,
        Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
        Option<Tensor<(P::Heads, usize, usize), E, D>>,
    )> {
        let residual = x.clone();
        let (attn_out, cache, mask) = self.mha.try_forward(
            self.rms_1.try_forward(x.clone().to_dtype())?.to_dtype(),
            layer,
            pos,
            pos_scale,
            pos_enc,
            cache,
            mask,
        )?;
        let x = attn_out + residual;
        let residual = x.clone();
        let mlp_out = self
            .mlp
            .try_forward(self.rms_2.try_forward(x.to_dtype())?.to_dtype())?;
        Ok((residual + mlp_out, cache, mask))
    }
}

pub struct Mistral<E: Dtype, P: Params, D1: Device<E>, D2: Device<E>>
where
    D1: Device<f32>,
    D2: Device<f32>,
{
    embedding: Embedding<P::Vocab, P::Hidden, E, D1>,
    pos_enc1: RotaryEmbedding<P::HeadDim, E, D1>,
    pos_enc2: RotaryEmbedding<P::HeadDim, E, D2>,
    blocks1: Vec<Block<E, P, D1>>,
    blocks2: Vec<Block<E, P, D2>>,
    ln: RmsNorm<P::Hidden, f32, D1>,
    lm: Linear<P::Hidden, P::Vocab, E, D1>,
    p: P,
}

fn load_block<E: Dtype, P: Params, D: Device<E>>(
    i: usize,
    p: P,
    dev: &D,
    loader: &SafeTensorLoader,
) -> Result<Block<E, P, D>>
where
    D: Device<f32>,
{
    let loader = loader.sub(format!("layers.{i}"));
    let in_ln = load_rmsnorm(dev, &loader.sub("input_layernorm"), P::RMS_NORM_EPS)?;
    let post_ln = load_rmsnorm(
        dev,
        &loader.sub("post_attention_layernorm"),
        P::RMS_NORM_EPS,
    )?;

    let k_proj = load_linear(dev, &loader.sub("self_attn.k_proj"))?;
    let q_proj = load_linear(dev, &loader.sub("self_attn.q_proj"))?;
    let v_proj = load_linear(dev, &loader.sub("self_attn.v_proj"))?;
    let o_proj = load_linear(dev, &loader.sub("self_attn.o_proj"))?;

    let down_proj = load_linear(dev, &loader.sub("mlp.down_proj"))?;
    let up_proj = load_linear(dev, &loader.sub("mlp.up_proj"))?;
    let gate_proj = load_linear(dev, &loader.sub("mlp.gate_proj"))?;

    Ok(Block {
        rms_1: in_ln,
        rms_2: post_ln,
        mha: MHA {
            k_proj,
            q_proj,
            v_proj,
            o_proj,
            p,
        },
        mlp: MLP {
            fc1: gate_proj,
            fc2: up_proj,
            proj: down_proj,
        },
    })
}
impl<E: Dtype, P: Params, D1: Device<E>, D2: Device<E>> Mistral<E, P, D1, D2>
where
    D1: Device<f32>,
    D2: Device<f32>,
{
    pub fn load_model(
        p: P,
        dev: &D1,
        dev2: &D2,
        loader: &SafeTensorLoader,
        split: usize,
    ) -> Result<Self> {
        let loader = loader.sub("model");
        let embedding = load_emedding(dev, &loader.sub("embed_tokens"))?;

        let mut blocks1 = Vec::new();
        let layers = p.layers().size();
        for i in 0..split {
            blocks1.push(load_block(i, p, dev, &loader)?);
        }

        let mut blocks2 = Vec::new();
        for i in split..layers {
            blocks2.push(load_block(i, p, dev2, &loader)?);
        }

        let pos_enc1 = RotaryEmbedding::new(dev, p.head_dim(), P::MAX_SEQ_LEN, P::ROE_BASE);
        let pos_enc2 = RotaryEmbedding::new(dev2, p.head_dim(), P::MAX_SEQ_LEN, P::ROE_BASE);

        let loader = loader.root();
        let ln = load_rmsnorm(dev, &loader.sub("model.norm"), P::RMS_NORM_EPS)?;
        let lm = load_linear(dev, &loader.sub("lm_head"))?;

        Ok(Self {
            embedding,
            blocks1,
            blocks2,
            ln,
            lm,
            pos_enc1,
            pos_enc2,
            p,
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq,), usize, D1>,
        pos: usize,
        pos_scale: usize,
        cache: (
            Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D1>>,
            Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D2>>,
        ),
    ) -> Result<(
        Tensor<(Seq, P::Vocab), E, D1>,
        (
            Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D1>>,
            Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D2>>,
        ),
    )> {
        let dev = x.dev().clone();
        let mut x = self.embedding.try_forward(x)?;
        let (mut cache1, mut cache2) = cache;
        let mut mask1 = None;
        for (i, b) in self.blocks1.iter().enumerate() {
            (x, cache1, mask1) =
                b.try_forward(x, i, pos, pos_scale, &self.pos_enc1, cache1, mask1)?;
        }

        let x = if self.blocks2.len() > 0 {
            let mut x = x.to_device(&D2::default());

            let mut mask2 = None;
            let base = self.blocks1.len();
            for (i, b) in self.blocks2.iter().enumerate() {
                (x, cache2, mask2) =
                    b.try_forward(x, base + i, pos, pos_scale, &self.pos_enc2, cache2, mask2)?;
            }

            x.to_device(&dev)
        } else {
            x
        };

        let x = self.ln.try_forward(x.to_dtype())?.to_dtype();
        let x = self.lm.try_forward(x)?;
        Ok((x, (cache1, cache2)))
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
    type KvHeads: ConstDim;
    type KvDim: ConstDim;
    type Layers: ConstDim;
    const MAX_SEQ_LEN: usize;
    const ROE_BASE: i64;
    const RMS_NORM_EPS: f64;

    fn vocab(&self) -> Self::Vocab;
    fn hidden(&self) -> Self::Hidden;
    fn mlp_dim(&self) -> Self::MlpDim;
    fn heads(&self) -> Self::Heads;
    fn head_dim(&self) -> Self::HeadDim;
    fn kv_heads(&self) -> Self::KvHeads;
    fn kv_dim(&self) -> Self::KvDim;
    fn layers(&self) -> Self::Layers;
}
