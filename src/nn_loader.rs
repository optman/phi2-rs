use crate::rmsnorm::RmsNorm;
use crate::tensor_loader::SafeTensorLoader;
use anyhow::Result;
use dfdx::nn::Embedding;
use dfdx::prelude::*;

pub(crate) fn load_emedding<Vocab: ConstDim, Model: ConstDim, E: Dtype, D: Device<E>>(
    dev: &D,
    loader: &SafeTensorLoader,
) -> Result<Embedding<Vocab, Model, E, D>>
where
    D: Device<f32>,
{
    let weight = loader.load_safetensor(dev, "weight")?;
    Ok(Embedding { weight })
}

pub(crate) fn load_rmsnorm<M: ConstDim, E: Dtype, D: Device<E>>(
    dev: &D,
    loader: &SafeTensorLoader,
) -> Result<RmsNorm<M, E, D>>
where
    D: Device<f32>,
{
    let gamma = loader.load_safetensor(dev, "weight")?;

    Ok(RmsNorm {
        gamma,
        epsilon: 1e-5,
    })
}
pub(crate) fn load_linear<I: ConstDim, O: ConstDim, E: Dtype, D: Device<E>>(
    dev: &D,
    loader: &SafeTensorLoader,
) -> Result<Linear<I, O, E, D>>
where
    D: Device<f32>,
{
    let weight = loader.load_safetensor(dev, "weight")?;
    let bias = dev.zeros();

    Ok(Linear { weight, bias })
}
