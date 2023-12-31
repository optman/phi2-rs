use dfdx::{prelude::*, tensor::Tensor, tensor_ops::Device};

pub trait Entry {
    type E;
}

pub struct Cache<KvHeads: Dim, HeadDim: Dim, Layers: Dim, E: Dtype, D: Device<E>> {
    //NOTE:my older gpu can't create zero size tensor, so use option to work around.
    pub k: Vec<Option<<Self as Entry>::E>>,
    pub v: Vec<Option<<Self as Entry>::E>>,

    pub cache_size: usize,
}

impl<KvHeads: Dim, HeadDim: Dim, Layers: Dim, E: Dtype, D: Device<E>> Entry
    for Cache<KvHeads, HeadDim, Layers, E, D>
{
    type E = Tensor<(usize, KvHeads, HeadDim), E, D>;
}

impl<KvHeads: Dim, HeadDim: Dim, Layers: Dim, E: Dtype, D: Device<E>>
    Cache<KvHeads, HeadDim, Layers, E, D>
{
    pub fn new(layers: Layers, cache_size: usize) -> Self {
        Self {
            k: vec![None; layers.size()],
            v: vec![None; layers.size()],
            cache_size,
        }
    }
    pub fn append(
        &mut self,
        layer: usize,
        k: <Self as Entry>::E,
        v: <Self as Entry>::E,
    ) -> (<Self as Entry>::E, <Self as Entry>::E) {
        let k = Self::_append(&mut self.k[layer], k, self.cache_size);
        let v = Self::_append(&mut self.v[layer], v, self.cache_size);

        (k, v)
    }

    fn _append(
        c: &mut Option<<Self as Entry>::E>,
        new_item: <Self as Entry>::E,
        max_cache_size: usize,
    ) -> <Self as Entry>::E {
        let cc = if let Some(mut cc) = Option::take(c) {
            if cc.shape().0.size() == max_cache_size - 1 {
                cc = cc.slice((1.., .., ..));
            }
            (cc, new_item).concat_tensor_along(Axis::<0>)
        } else {
            new_item
        };

        *c = Some(cc.clone());
        cc
    }
}
