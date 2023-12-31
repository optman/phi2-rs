use crate::tensor_loader::SafeTensorLoader;
use anyhow::Result;
use dfdx::dtypes::f16;
use dfdx::nn::{Embedding, LayerNorm1D};
use dfdx::prelude::*;

pub(crate) fn load_emedding<Vocab: ConstDim, Model: ConstDim, E: Dtype, D: Device<E>>(
    dev: &D,
    loader: &SafeTensorLoader,
) -> Result<Embedding<Vocab, Model, E, D>>
where
    D: Device<f16> + ToDtypeKernel<f16, E>,
{
    let weight = loader.load_safetensor(dev, "weight")?;
    Ok(Embedding { weight })
}

pub(crate) fn load_layernorm<M: ConstDim, E: Dtype, D: Device<E>>(
    dev: &D,
    loader: &SafeTensorLoader,
) -> Result<LayerNorm1D<M, E, D>>
where
    D: Device<f16> + ToDtypeKernel<f16, E>,
{
    let gamma = loader.load_safetensor(dev, "weight")?;
    let beta = loader.load_safetensor(dev, "bias")?;

    Ok(LayerNorm1D {
        gamma,
        beta,
        epsilon: 1e-5,
    })
}
pub(crate) fn load_linear<I: ConstDim, O: ConstDim, E: Dtype, D: Device<E>>(
    dev: &D,
    loader: &SafeTensorLoader,
) -> Result<Linear<I, O, E, D>>
where
    D: Device<f16> + ToDtypeKernel<f16, E>,
{
    let weight = loader.load_safetensor(dev, "weight")?;
    let bias = loader.load_safetensor(dev, "bias")?;

    Ok(Linear { weight, bias })
}
