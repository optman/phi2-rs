use crate::rmsnorm::RmsNorm;
use crate::tensor_loader::SafeTensorLoader;
use anyhow::Result;
use dfdx::nn::Embedding;
use dfdx::prelude::*;

pub(crate) fn load_emedding<Vocab: ConstDim, Model: ConstDim, E: Dtype, D: Device<E>>(
    dev: &D,
    loader: &SafeTensorLoader,
) -> Result<Embedding<Vocab, Model, E, D>> {
    let weight = loader.load_safetensor(dev, "weight")?;
    Ok(Embedding { weight })
}

pub(crate) fn load_rmsnorm<M: ConstDim, E: Dtype, D: Device<E>>(
    dev: &D,
    loader: &SafeTensorLoader,
    eps: f64,
) -> Result<RmsNorm<M, E, D>> {
    let gamma = loader.load_safetensor(dev, "weight")?;

    Ok(RmsNorm {
        gamma,
        epsilon: eps,
    })
}
pub(crate) fn load_linear<I: ConstDim, O: ConstDim, E: Dtype, D: Device<E>>(
    dev: &D,
    loader: &SafeTensorLoader,
) -> Result<Linear<I, O, E, D>> {
    let weight = loader.load_safetensor(dev, "weight")?;
    let bias = match loader.load_safetensor(dev, "bias") {
        Ok(bias) => bias,
        Err(_) => dev.zeros(),
    };

    Ok(Linear { weight, bias })
}
