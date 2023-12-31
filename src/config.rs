use crate::model::Params;
use dfdx::prelude::*;
#[derive(Debug, Clone, Copy)]
pub struct ConfigV2 {}

impl Params for ConfigV2 {
    type Vocab = Const<51200>;
    type Hidden = Const<2560>;
    type MlpDim = Const<10240>;
    type Heads = Const<32>;
    type HeadDim = Const<80>;
    type QkvDim = Const<7680>;
    type Layers = Const<32>;
    type RoeDim = Const<32>;

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

    fn qkv_dim(&self) -> Self::QkvDim {
        Self::QkvDim {}
    }

    fn layers(&self) -> Self::Layers {
        Self::Layers {}
    }

    fn roe_dim(&self) -> Self::RoeDim {
        Self::RoeDim {}
    }
}
