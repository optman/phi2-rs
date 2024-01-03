use ::safetensors::tensor::{SafeTensorError, SafeTensors};
use anyhow::{Error, Result};
use dfdx::prelude::*;
use half::bf16;
use std::rc::Rc;
use std::vec::Vec;

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
    ) -> Result<Tensor<S, E, D>> {
        let mut paths = self.prefixs.clone();
        paths.push(path.to_owned());
        let full_path = paths.join(".");

        let mut t: Tensor<_, E, _> = dev.zeros();
        let mut err = None;
        for ts in &*self.tensors {
            match load_safetensor_bf16(&mut t, &ts.get().0, &full_path) {
                Ok(_) => {
                    return Ok(t);
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

fn load_safetensor_bf16<S: Shape, E: Dtype, D: CopySlice<E>>(
    t: &mut Tensor<S, E, D>,
    tensors: &SafeTensors,
    key: &str,
) -> Result<(), SafeTensorError> {
    let tensor_view = tensors.tensor(key)?;
    let v = tensor_view.data();
    assert_eq!(
        tensor_view.shape(),
        t.shape().concrete().into(),
        "SafeTensors shape did not match tensor shape"
    );
    let mut c = Vec::with_capacity(v.len() / 2);
    let mut i = 0;
    while i < v.len() {
        let value: f32 = bf16::from_le_bytes([v[i], v[i + 1]]).to_f32();
        c.push(E::from_f32(value).unwrap());
        i += 2;
    }
    t.copy_from(&c);
    Ok(())
}
