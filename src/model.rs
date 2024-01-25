use crate::cache::Cache;
use crate::nn_loader::*;
use crate::rmsnorm::RmsNorm;
use crate::rotary::RotaryEmbedding;
use crate::tensor_loader::SafeTensorLoader;
use anyhow::Result;
use dfdx::dtypes::{f16, AMP};
use dfdx::prelude::*;
use itertools::Itertools;
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
        cache: &mut Option<&mut Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
        mask: &mut Option<Tensor<(P::Heads, usize, usize), E, D>>,
    ) -> Result<Tensor<(Seq, P::Hidden), E, D>, Error> {
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

        let mask = {
            let _mask = match mask {
                Some(mask) => mask.clone(),
                None => {
                    let attn_seq = kv_seq;
                    let mask = dev.upper_tri_like(&(attn_seq, attn_seq), E::neg_infinity(), 1);
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
            .try_reshape_like(&(seq, hidden))?
            .realize();

        let out = self.o_proj.try_forward(v)?;

        Ok(out)
    }
}
struct Expert<E: Dtype, P: Params, D: Device<E>> {
    w1: Linear<P::Hidden, P::MlpDim, E, D>,
    w2: Linear<P::MlpDim, P::Hidden, E, D>,
    w3: Linear<P::Hidden, P::MlpDim, E, D>,
}
impl<E: Dtype, P: Params, D: Device<E>> Expert<E, P, D> {
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::Hidden), E, D>,
    ) -> Result<Tensor<(Seq, P::Hidden), E, D>> {
        let y1 = self.w1.try_forward(x.clone())?;
        let y1 = y1.clone().sigmoid() * y1;
        let y2 = self.w3.try_forward(x)?;
        let y = self.w2.try_forward(y1 * y2)?;
        Ok(y)
    }
}

struct SpareMoeBlock<E: Dtype, P: Params, D: Device<E>> {
    gate: Linear<P::Hidden, P::GateDim, E, D>,
    experts: Vec<Expert<E, P, D>>,
}
impl<E: Dtype, P: Params, D: Device<E>> SpareMoeBlock<E, P, D> {
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::Hidden), E, D>,
    ) -> Result<Tensor<(Seq, P::Hidden), E, D>> {
        let weights = self.gate.try_forward(x.clone())?;
        let (seq_len, loc_exp) = weights.shape();
        let (seq_len, loc_exp) = (seq_len.size(), loc_exp.size());
        let weights = weights.softmax::<Axis<1>>().as_vec();

        let dev = x.dev();
        let mut sew = Vec::with_capacity(seq_len * P::NUM_EXPERTS_PER_TOK);
        let mut es = std::iter::repeat(vec![]).take(loc_exp).collect::<Vec<_>>();

        for (seq, idx_w) in weights.into_iter().chunks(loc_exp).into_iter().enumerate() {
            let mut idx_w = idx_w.enumerate().collect::<Vec<_>>();
            idx_w.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            let sum = idx_w
                .iter()
                .take(P::NUM_EXPERTS_PER_TOK)
                .map(|(_, w)| w.to_f32().unwrap())
                .sum::<f32>();
            for i in 0..P::NUM_EXPERTS_PER_TOK {
                let (e, w) = idx_w[i];
                sew.push((e, w / E::from_f32(sum).unwrap()));
                es[e].push(seq);
            }
        }

        let mut ys = Vec::with_capacity(loc_exp);
        for i in 0..loc_exp {
            let xs = &es[i];
            if xs.len() == 0 {
                ys.push(None);
                continue;
            }
            let idx = dev.tensor_from_vec(xs.clone(), (xs.len(),));
            let xs = x.clone().gather(idx);
            ys.push(Some(self.experts[i].try_forward(xs)?));
        }

        let mut idx = vec![0; loc_exp];
        let mut result = Vec::with_capacity(seq_len);
        for rs in &sew.into_iter().chunks(P::NUM_EXPERTS_PER_TOK) {
            let mut sum = None;
            for (e, w) in rs {
                let i = idx[e];
                idx[e] = i + 1;
                let i = dev.tensor(i);
                let y = ys[e].clone().unwrap().select(i);
                let y = y * w.to_f32().unwrap();
                sum = match sum {
                    Some(old) => Some(old + y),
                    None => Some(y),
                };
            }

            result.push(sum.unwrap());
        }

        Ok(result.stack().realize())
    }
}
struct Block<E: Dtype, P: Params, D: Device<E>>
where
    D: Device<f32>,
{
    rms_1: RmsNorm<P::Hidden, f32, D>,
    rms_2: RmsNorm<P::Hidden, f32, D>,
    mha: MHA<E, P, D>,
    block_spare_moe: SpareMoeBlock<E, P, D>,
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
        cache: &mut Option<&mut Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
        mask: &mut Option<Tensor<(P::Heads, usize, usize), E, D>>,
    ) -> Result<Tensor<(Seq, P::Hidden), E, D>> {
        let residual = x.clone();
        let attn_out = self.mha.try_forward(
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
            .block_spare_moe
            .try_forward(self.rms_2.try_forward(x.to_dtype())?.to_dtype())?;
        Ok(mlp_out + residual)
    }
}

pub struct Mixtral<E: Dtype, P: Params, D1: Device<E>, D2: Device<E>>
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

    let attn_l = loader.sub("self_attn");
    let k_proj = load_linear(dev, &attn_l.sub("k_proj"))?;
    let q_proj = load_linear(dev, &attn_l.sub("q_proj"))?;
    let v_proj = load_linear(dev, &attn_l.sub("v_proj"))?;
    let o_proj = load_linear(dev, &attn_l.sub("o_proj"))?;

    let moe_l = loader.sub("block_sparse_moe");
    let gate = load_linear(dev, &moe_l.sub("gate"))?;

    let exps_l = moe_l.sub("experts");
    let mut experts = Vec::new();
    for i in 0..P::NUM_LOCAL_EXPERTS {
        let exp_l = exps_l.sub(format!("{i}"));
        let w1 = load_linear(dev, &exp_l.sub("w1"))?;
        let w2 = load_linear(dev, &exp_l.sub("w2"))?;
        let w3 = load_linear(dev, &exp_l.sub("w3"))?;
        experts.push(Expert { w1, w2, w3 });
    }

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
        block_spare_moe: SpareMoeBlock { gate, experts },
    })
}
impl<E: Dtype, P: Params, D1: Device<E>, D2: Device<E>> Mixtral<E, P, D1, D2>
where
    D1: Device<f32>,
    D2: Device<f32>,
{
    pub fn load_model(
        p: P,
        dev: &D1,
        dev2: &D2,
        loader: &SafeTensorLoader,
        split: Option<usize>,
    ) -> Result<Self> {
        let loader = loader.sub("model");
        let embedding = load_emedding(dev, &loader.sub("embed_tokens"))?;

        let mut blocks1 = Vec::new();
        let mut blocks2 = Vec::new();
        let layers = p.layers().size();
        if let Some(split) = split {
            for i in 0..split {
                blocks1.push(load_block(i, p, dev, &loader)?);
            }

            for i in split..layers {
                blocks2.push(load_block(i, p, dev2, &loader)?);
            }
        } else {
            for i in 0..layers {
                blocks1.push(load_block(i, p, dev, &loader)?);
            }
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
        cache: &mut (
            &mut Option<&mut Cache<P::KvHeads, P::HeadDim, P::Layers, E, D1>>,
            &mut Option<&mut Cache<P::KvHeads, P::HeadDim, P::Layers, E, D2>>,
        ),
    ) -> Result<Tensor<(Seq, P::Vocab), E, D1>> {
        let dev = x.dev().clone();
        let mut x = self.embedding.try_forward(x)?;
        let (cache1, cache2) = (&mut cache.0, &mut cache.1);
        let mut mask1 = None;
        for (i, b) in self.blocks1.iter().enumerate() {
            x = b.try_forward(x, i, pos, pos_scale, &self.pos_enc1, cache1, &mut mask1)?;
        }

        let x = if self.blocks2.len() > 0 {
            let mut x = x.to_device(&D2::default());

            let mut mask2 = None;
            let base = self.blocks1.len();
            for (i, b) in self.blocks2.iter().enumerate() {
                x = b.try_forward(
                    x,
                    base + i,
                    pos,
                    pos_scale,
                    &self.pos_enc2,
                    cache2,
                    &mut mask2,
                )?;
            }

            x.to_device(&dev)
        } else {
            x
        };

        let x = self.ln.try_forward(x.to_dtype())?.to_dtype();
        let x = self.lm.try_forward(x)?;
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
    type KvHeads: ConstDim;
    type KvDim: ConstDim;
    type Layers: ConstDim;
    type GateDim: ConstDim;
    const MAX_SEQ_LEN: usize;
    const ROE_BASE: i64;
    const RMS_NORM_EPS: f64;
    const NUM_EXPERTS_PER_TOK: usize;
    const NUM_LOCAL_EXPERTS: usize;

    fn vocab(&self) -> Self::Vocab;
    fn hidden(&self) -> Self::Hidden;
    fn mlp_dim(&self) -> Self::MlpDim;
    fn heads(&self) -> Self::Heads;
    fn head_dim(&self) -> Self::HeadDim;
    fn kv_heads(&self) -> Self::KvHeads;
    fn kv_dim(&self) -> Self::KvDim;
    fn layers(&self) -> Self::Layers;
}
