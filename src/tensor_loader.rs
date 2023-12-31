use ::safetensors::SafeTensorError;
use anyhow::{Error, Result};
use dfdx::dtypes::f16;
use dfdx::prelude::*;
use dfdx::safetensors::SafeTensors;
use std::rc::Rc;

#[derive(yoke::Yokeable)]
struct SafeTensors_<'a>(SafeTensors<'a>);

pub struct SafeTensorLoader {
    tensors: Rc<Vec<yoke::Yoke<SafeTensors_<'static>, memmap2::Mmap>>>,
    prefixs: Vec<String>,
}

impl SafeTensorLoader {
    pub(crate) fn new(paths: Vec<String>) -> Result<Self> {
        let mut tensors = Vec::with_capacity(paths.len());
        for p in paths {
            let f = std::fs::File::open(&p).map_err(|e| Error::msg(format!("{e} {p}")))?;
            let buffer = unsafe { memmap2::MmapOptions::new().map(&f)? };
            let t = yoke::Yoke::<SafeTensors_<'static>, memmap2::Mmap>::try_attach_to_cart(
                buffer,
                |buffer: &[u8]| {
                    let st = SafeTensors::deserialize(buffer)?;
                    Ok::<SafeTensors_<'_>, SafeTensorError>(SafeTensors_(st))
                },
            )?;
            tensors.push(t);
        }

        Ok(Self {
            tensors: Rc::new(tensors),
            prefixs: Vec::new(),
        })
    }

    pub(crate) fn sub<S: AsRef<str>>(&self, path: S) -> Self {
        let mut prefixs = self.prefixs.clone();
        prefixs.push(path.as_ref().to_owned());
        Self {
            tensors: self.tensors.clone(),
            prefixs,
        }
    }

    pub(crate) fn root(&self) -> Self {
        Self {
            tensors: self.tensors.clone(),
            prefixs: Vec::new(),
        }
    }

    pub(crate) fn load_safetensor<S: ConstShape, E: Dtype, D: Device<E>>(
        &self,
        dev: &D,
        path: &str,
    ) -> Result<Tensor<S, E, D>>
    where
        D: Device<f16> + ToDtypeKernel<f16, E>,
    {
        let mut paths = self.prefixs.clone();
        paths.push(path.to_owned());
        let full_path = paths.join(".");

        let mut t: Tensor<_, f16, _> = dev.zeros();
        let mut err = None;
        for ts in &*self.tensors {
            match t.load_safetensor(&ts.get().0, &full_path) {
                Ok(_) => {
                    return Ok(t.to_dtype::<E>());
                }
                Err(e @ SafeTensorError::TensorNotFound(_)) => {
                    err = Some(Error::from(e));
                    continue;
                }
                Err(e) => return Err(Error::from(e)),
            }
        }

        match err {
            None => unreachable!(),
            Some(err) => Err(err),
        }
    }
}
