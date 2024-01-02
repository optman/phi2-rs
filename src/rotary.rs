use dfdx::prelude::*;

#[derive(Clone, Debug, Default)]
pub struct RotaryEmbeddingConfig<RotaryDim: Dim> {
    pub rotary_dim: RotaryDim,
    pub max_seq: usize,
    pub base: i64,
}

impl<RotaryDim: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D>
    for RotaryEmbeddingConfig<RotaryDim>
{
    type Built = RotaryEmbedding<RotaryDim, E, D>;

    fn try_build_on_device(&self, dev: &D) -> Result<Self::Built, dfdx_core::tensor::Error> {
        Ok(RotaryEmbedding::new(
            dev,
            self.rotary_dim,
            self.max_seq,
            self.base,
        ))
    }
}

#[allow(clippy::type_complexity)]
#[derive(Clone, Debug, ZeroGrads, UpdateParams, ResetParams, SaveSafeTensors, LoadSafeTensors)]
pub struct RotaryEmbedding<RotaryDim: Dim, E: Dtype, D: Device<E>> {
    rotary_dim: RotaryDim,
    cos: Tensor<(usize, RotaryDim), E, D>,
    sin: Tensor<(usize, RotaryDim), E, D>,
}
impl<RotaryDim: Dim, E: Dtype, D: Device<E>> RotaryEmbedding<RotaryDim, E, D> {
    #[allow(clippy::type_complexity)]
    pub fn try_forward<Seq: Dim, Headers: Dim, HeadDim: Dim>(
        &self,
        x: Tensor<(Seq, Headers, HeadDim), E, D>,
        pos: usize,
        pos_scale: usize,
    ) -> Result<Tensor<(Seq, Headers, HeadDim), E, D>, Error> {
        let (seq, _header, head_dim) = *x.shape();
        let rotary_dim = self.rotary_dim.size();
        let (x_rot, x_pass) = if rotary_dim < head_dim.size() {
            (
                x.clone().slice((.., .., ..rotary_dim)).realize(),
                Some(x.slice((.., .., rotary_dim..))),
            )
        } else {
            (x.realize(), None)
        };

        let half_rotary_dim = rotary_dim / 2;
        let first_half = x_rot.clone().slice((.., .., ..half_rotary_dim));
        let second_half = x_rot.clone().slice((.., .., half_rotary_dim..rotary_dim));

        let idx = x_rot.dev().tensor_from_vec(
            (pos..pos + seq.size()).map(|n| n / pos_scale).collect(),
            (seq,),
        );

        let sub_cos: Tensor<(Seq, RotaryDim), _, _> =
            self.cos.clone().gather(idx.clone()).realize();
        let sub_sin: Tensor<(Seq, RotaryDim), _, _> = self.sin.clone().gather(idx).realize();

        let neg_half_x: Tensor<(Seq, Headers, RotaryDim), _, _> =
            (first_half.negate(), second_half)
                .concat_tensor_along(Axis::<2>)
                .realize();

        let y: Tensor<(Seq, Headers, usize), _, _> = (sub_sin.broadcast_like(&neg_half_x)
            * neg_half_x
            + sub_cos.broadcast_like(&x_rot) * x_rot)
            .realize();

        let y = match x_pass {
            Some(x_pass) => (y, x_pass).concat_tensor_along(Axis::<2>).realize(),
            None => y,
        };

        Ok(y.realize())
    }

    pub fn new(dev: &D, rotary_dim: RotaryDim, max_seq: usize, base: i64) -> Self {
        let half_rotary_dim = rotary_dim.size() / 2;
        let theda = dev.tensor_from_vec(
            (0..rotary_dim.size())
                .step_by(2)
                .map(|c| E::from_usize(c).unwrap())
                .collect(),
            (half_rotary_dim,),
        );
        let theda = ((theda / rotary_dim.size() as f32) * (base as f32).ln())
            .exp()
            .recip();

        let idx_theda: Tensor<(usize, usize), E, D> = (0..max_seq)
            .map(|i| theda.clone() * i as f32)
            .collect::<Vec<_>>()
            .stack()
            .realize();

        let idx_theda = (idx_theda.clone(), idx_theda)
            .concat_tensor_along(Axis::<1>)
            .realize();

        let cos = idx_theda.clone().cos();
        let sin = idx_theda.sin();

        Self {
            rotary_dim,
            cos,
            sin,
        }
    }
}
