use crate::model::Params;
use dfdx::prelude::*;
#[derive(Debug, Clone, Copy)]
pub struct ConfigV2 {}

impl Params for ConfigV2 {
    type Vocab = Const<32000>;
    type Hidden = Const<2048>;
    type MlpDim = Const<5632>;
    type Heads = Const<32>;
    type HeadDim = Const<64>;
    type KvHeads = Const<4>;
    type KvDim = Const<256>;
    type Layers = Const<22>;

    const MAX_SEQ_LEN: usize = 4096;
    const ROE_BASE: i64 = 10000;

    fn vocab(&self) -> Self::Vocab {
        Self::Vocab {}
    }

    fn hidden(&self) -> Self::Hidden {
        Self::Hidden {}
    }

    fn mlp_dim(&self) -> Self::MlpDim {
        Self::MlpDim {}
    }

    fn heads(&self) -> Self::Heads {
        Self::Heads {}
    }

    fn head_dim(&self) -> Self::HeadDim {
        Self::HeadDim {}
    }

    fn kv_heads(&self) -> Self::KvHeads {
        Self::KvHeads {}
    }

    fn kv_dim(&self) -> Self::KvDim {
        Self::KvDim {}
    }

    fn layers(&self) -> Self::Layers {
        Self::Layers {}
    }
}
