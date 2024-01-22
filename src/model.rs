use crate::nn_loader::*;
use crate::rmsnorm::RmsNorm;
use crate::tensor_loader::SafeTensorLoader;
use anyhow::Result;
use dfdx::dtypes::{f16, AMP};
use dfdx::prelude::*;
use std::fmt::Debug;

pub trait Dtype: dfdx::prelude::Dtype + num_traits::Float {}
impl Dtype for f32 {}
impl Dtype for f16 {}
impl Dtype for AMP<f16> {}

struct MambaBlock<E: Dtype, P: Params, D: Device<E>> {
    in_proj: Linear<P::DModel, P::DInnerX2, E, D>,
    conv1d: (
        //(OutChan, InChan/Group, Kernel)
        Tensor<(P::DInner, Const<1>, P::DConv), E, D>,
        //(OutChan,)
        Tensor<(P::DInner,), E, D>,
    ),
    x_proj: Linear<P::DInner, P::XProjO, E, D>,
    dt_proj: Linear<P::DtRank, P::DInner, E, D>,
    a_log: Tensor<(P::DInner, P::DState), E, D>,
    d: Tensor<(P::DInner,), E, D>,
    out_proj: Linear<P::DInner, P::DModel, E, D>,
    p: P,
}
impl<E: Dtype, P: Params, D: Device<E>> MambaBlock<E, P, D> {
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::DModel), E, D>,
    ) -> Result<Tensor<(Seq, P::DModel), E, D>, Error> {
        let (d_inner, d_conv) = (self.p.d_inner().size(), self.p.d_conv().size());
        let seq = x.shape().0.size();
        let xs_and_res = self.in_proj.try_forward(x)?;
        let xs: Tensor<(usize, usize), _, _> = xs_and_res.clone().slice((.., ..d_inner)).realize();
        let res: Tensor<(Seq, P::DInner), _, _> = xs_and_res.slice((.., d_inner..)).realize();

        let xs = xs.permute::<_, Axes2<1, 0>>();
        let weight: Tensor<(usize, usize, usize), _, _> = self.conv1d.0.clone().realize();
        let xs: Tensor<(P::DInner, usize), _, _> =
            (xs, weight).conv1d(1, d_conv - 1, 1, d_inner).realize();
        let xs = xs.clone() + self.conv1d.1.clone().broadcast_like(&xs);
        let xs = xs.slice((.., ..seq));
        let xs: Tensor<(Seq, P::DInner), _, _> = xs.permute::<_, Axes2<1, 0>>().realize();
        let xs = xs.clone().sigmoid() * xs;
        let res = res.clone().sigmoid() * res;
        let ys = self.ssm(xs)? * res;
        self.out_proj.try_forward(ys)
    }
    pub fn ssm<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::DInner), E, D>,
    ) -> Result<Tensor<(Seq, P::DInner), E, D>, Error> {
        let n = self.p.d_state().size();
        let dt_rank = self.p.dt_rank().size();
        let a = self.a_log.clone().exp().negate();
        let d = self.d.clone();
        let x_db1 = self.x_proj.try_forward(x.clone())?;
        let delta: Tensor<(Seq, P::DtRank), _, _> = x_db1.clone().slice((.., ..dt_rank)).realize();
        let b: Tensor<(Seq, P::DState), _, _> =
            x_db1.clone().slice((.., dt_rank..dt_rank + n)).realize();
        let c: Tensor<(Seq, P::DState), _, _> = x_db1.clone().slice((.., dt_rank + n..)).realize();
        let delta = self.dt_proj.try_forward(delta)?;
        let delta = (delta.exp() + 1.0).ln();
        let ss = self.selective_scan(x, delta, a, b, c, d)?;
        Ok(ss)
    }

    pub fn selective_scan<Seq: Dim>(
        &self,
        u: Tensor<(Seq, P::DInner), E, D>,
        delta: Tensor<(Seq, P::DInner), E, D>,
        a: Tensor<(P::DInner, P::DState), E, D>,
        b: Tensor<(Seq, P::DState), E, D>,
        c: Tensor<(Seq, P::DState), E, D>,
        d: Tensor<(P::DInner,), E, D>,
    ) -> Result<Tensor<(Seq, P::DInner), E, D>, Error> {
        let dev = u.dev().clone();
        let (l, d_in) = *u.shape();
        let n = a.shape().1;
        let delta = delta.broadcast_like(&(l, d_in, n));
        let delta_a = (delta.clone() * a.broadcast_like(&(l, d_in, n))).exp();
        let delta_b_u =
            delta * b.broadcast_like(&(l, d_in, n)) * (u.clone().broadcast_like(&(l, d_in, n)));

        let mut xs = dev.zeros_like(&(d_in, n));
        let mut ys = Vec::with_capacity(l.size());
        for i in 0..l.size() {
            xs = delta_a.clone().select(dev.tensor(i)) * xs
                + delta_b_u.clone().select(dev.tensor(i));
            let y = xs
                .clone()
                .matmul(c.clone().select(dev.tensor(i)))
                .reshape_like(&(d_in,));
            ys.push(y);
        }
        let ys = ys.stack().realize();

        Ok(ys + u * d.broadcast_like(&(l, d_in)))
    }
}

struct ResidualBlock<E: Dtype, P: Params, D: Device<E>> {
    ln: RmsNorm<P::DModel, E, D>,
    mixer: MambaBlock<E, P, D>,
}

impl<E: Dtype, P: Params, D: Device<E>> ResidualBlock<E, P, D> {
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::DModel), E, D>,
    ) -> Result<Tensor<(Seq, P::DModel), E, D>> {
        let y = self.ln.try_forward(x.clone())?;
        let y = self.mixer.try_forward(y)?;
        Ok(x + y)
    }
}

pub struct Mamba<E: Dtype, P: Params, D: Device<E>> {
    embedding: Embedding<P::Vocab, P::DModel, E, D>,
    blocks: Vec<ResidualBlock<E, P, D>>,
    ln: RmsNorm<P::DModel, E, D>,
    lm: Linear<P::DModel, P::Vocab, E, D>,
    p: P,
}

impl<E: Dtype, P: Params, D: Device<E>> Mamba<E, P, D> {
    pub fn load_model(p: P, dev: &D, loader: &SafeTensorLoader) -> Result<Self> {
        let loader = loader.sub("backbone");
        let ln = load_rmsnorm(dev, &loader.sub("norm_f"), P::RMS_NORM_EPS)?;
        let embedding = load_emedding(dev, &loader.sub("embedding"))?;
        let lm = Linear {
            weight: embedding.weight.clone(),
            bias: dev.zeros(),
        };

        let mut blocks = Vec::new();

        for i in 0..P::LAYERS {
            let loader = loader.sub(format!("layers.{i}"));
            let ln = load_rmsnorm(dev, &loader.sub("norm"), P::RMS_NORM_EPS)?;
            let loader = loader.sub("mixer");
            let in_proj = load_linear(dev, &loader.sub("in_proj"))?;
            let x_proj = load_linear(dev, &loader.sub("x_proj"))?;
            let dt_proj = load_linear(dev, &loader.sub("dt_proj"))?;
            let out_proj = load_linear(dev, &loader.sub("out_proj"))?;
            let a_log = loader.load_safetensor(dev, "A_log")?;
            let d = loader.load_safetensor(dev, "D")?;

            let conv1d = (
                loader.load_safetensor(dev, "conv1d.weight")?,
                loader.load_safetensor(dev, "conv1d.bias")?,
            );

            blocks.push(ResidualBlock {
                ln,
                mixer: MambaBlock {
                    in_proj,
                    x_proj,
                    dt_proj,
                    out_proj,
                    a_log,
                    d,
                    conv1d,
                    p,
                },
            });
        }

        Ok(Self {
            embedding,
            blocks,
            ln,
            lm,
            p,
        })
    }

    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq,), usize, D>,
    ) -> Result<Tensor<(P::Vocab,), E, D>> {
        let dev = x.dev().clone();
        let seq = x.shape().0.size();
        let mut x = self.embedding.try_forward(x)?;
        for b in self.blocks.iter() {
            x = b.try_forward(x)?;
        }
        let x = x.gather(dev.tensor_from_vec(vec![seq - 1], (1,)));
        let x = self.ln.try_forward(x)?;
        let x = self.lm.try_forward(x)?;
        Ok(x.reshape_like(&(self.p.vocab(),)))
    }

    #[allow(dead_code)]
    pub fn params(&self) -> &P {
        &self.p
    }
}

pub trait Params: Debug + Clone + Copy {
    type Vocab: ConstDim;
    type DModel: ConstDim;
    type DtRank: ConstDim;
    type DConv: ConstDim;
    type DState: ConstDim;
    type DInner: ConstDim;
    type DInnerX2: ConstDim;
    type XProjO: ConstDim;
    const MAX_SEQ_LEN: usize;
    const RMS_NORM_EPS: f64;
    const LAYERS: usize;

    fn vocab(&self) -> Self::Vocab;
    fn d_model(&self) -> Self::DModel;
    fn dt_rank(&self) -> Self::DtRank;
    fn d_conv(&self) -> Self::DConv;
    fn d_state(&self) -> Self::DState;
    fn d_inner(&self) -> Self::DInner;
    fn d_inner_x2(&self) -> Self::DInnerX2;
    fn x_proj_o(&self) -> Self::XProjO;
}
