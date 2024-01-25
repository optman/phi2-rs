use dfdx::{prelude::*, tensor::Tensor, tensor_ops::Device};

pub trait Entry {
    type ConvState;
    type SSMState;
}

pub struct Cache<DInner: Dim, DState: Dim, E: Dtype, D: Device<E>> {
    pub conv_state: Vec<Option<<Self as Entry>::ConvState>>,
    pub ssm_state: Vec<Option<<Self as Entry>::SSMState>>,
}

impl<DInner: Dim, DState: Dim, E: Dtype, D: Device<E>> Entry for Cache<DInner, DState, E, D> {
    type ConvState = Tensor<(DInner, usize), E, D>;
    type SSMState = Tensor<(DInner, DState), E, D>;
}

impl<DInner: Dim, DState: Dim, E: Dtype, D: Device<E>> Cache<DInner, DState, E, D> {
    pub fn new(layers: usize) -> Self {
        Self {
            conv_state: vec![None; layers],
            ssm_state: vec![None; layers],
        }
    }
}
