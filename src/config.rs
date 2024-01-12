use crate::model::Params;
use dfdx::prelude::*;
#[derive(Debug, Clone, Copy)]
pub struct ConfigV2 {}

impl Params for ConfigV2 {
    type Vocab = Const<32000>;
    type Hidden = Const<4096>;
    type MlpDim = Const<14336>;
    type Heads = Const<32>;
    type HeadDim = Const<128>;
    type KvHeads = Const<8>;
    type KvDim = Const<1024>;
    type Layers = Const<32>;
    type GateDim = Const<2>;

    const MAX_SEQ_LEN: usize = 32768;
    const ROE_BASE: i64 = 10000;
    const RMS_NORM_EPS: f64 = 1e-5;
    const NUM_EXPERTS_PER_TOK: usize = 2;
    const NUM_LOCAL_EXPERTS: usize = 2;

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
