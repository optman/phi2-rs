use dfdx::prelude::*;

pub struct RmsNorm<M: Dim, E: Dtype, D: Device<E>> {
    pub gamma: Tensor<(M,), E, D>,
    pub epsilon: f64,
}
impl<M: Dim, E: Dtype, D: Device<E>> RmsNorm<M, E, D> {
    pub fn try_forward<Batch: Dim>(
        &self,
        x: Tensor<(Batch, M), E, D>,
    ) -> Result<Tensor<(Batch, M), E, D>, Error> {
        let shape = *x.shape();
        let var = x.clone().square().mean::<_, Axis<1>>();
        let inv_std = (var + self.epsilon).sqrt().recip();
        let y = inv_std.broadcast_like(&shape) * x;
        let y = self.gamma.clone().broadcast_like(&shape) * y;
        Ok(y)
    }
}
