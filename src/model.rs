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
struct Block<E: Dtype, P: Params, D: Device<E>> {
    rms_1: RmsNorm<P::Hidden, E, D>,
    rms_2: RmsNorm<P::Hidden, E, D>,
    mha: MHA<E, P, D>,
    mlp: MLP<E, P, D>,
}

#[allow(clippy::type_complexity)]
impl<E: Dtype, P: Params, D: Device<E>> Block<E, P, D> {
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
            self.rms_1.try_forward(x.clone())?,
            layer,
            pos,
            pos_scale,
            pos_enc,
            cache,
            mask,
        )?;
        let x = attn_out + residual;
        let residual = x.clone();
        let mlp_out = self.mlp.try_forward(self.rms_2.try_forward(x)?)?;
        Ok((residual + mlp_out, cache, mask))
    }
}

pub struct PhiLM<E: Dtype, P: Params, D: Device<E>> {
    embedding: Embedding<P::Vocab, P::Hidden, E, D>,
    blocks: Vec<Block<E, P, D>>,
    ln: RmsNorm<P::Hidden, E, D>,
    lm: Linear<P::Hidden, P::Vocab, E, D>,
    pos_enc: RotaryEmbedding<P::HeadDim, E, D>,
    p: P,
}

impl<E: Dtype, P: Params, D: Device<E>> PhiLM<E, P, D> {
    pub fn load_model(p: P, dev: &D, loader: &SafeTensorLoader) -> Result<Self> {
        let loader = loader.sub("model");
        let ln = load_rmsnorm(dev, &loader.sub("norm"), P::RMS_NORM_EPS)?;
        let embedding = load_emedding(dev, &loader.sub("embed_tokens"))?;

        let mut blocks = Vec::new();

        for i in 0..p.layers().size() {
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

            blocks.push(Block {
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
            });
        }

        let loader = loader.root();
        let lm = load_linear(dev, &loader.sub("lm_head"))?;

        let pos_enc = RotaryEmbedding::new(dev, p.head_dim(), P::MAX_SEQ_LEN, P::ROE_BASE);

        Ok(Self {
            embedding,
            blocks,
            ln,
            lm,
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
        mut cache: Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
    ) -> Result<(
        Tensor<(Seq, P::Vocab), E, D>,
        Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
    )> {
        let mut x = self.embedding.try_forward(x)?;
        let mut mask = None;
        for (i, b) in self.blocks.iter().enumerate() {
            (x, cache, mask) = b.try_forward(x, i, pos, pos_scale, &self.pos_enc, cache, mask)?;
        }
        let x = self.ln.try_forward(x)?;
        let x = self.lm.try_forward(x)?;
        Ok((x, cache))
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
