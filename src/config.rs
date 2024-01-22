use crate::model::Params;
use dfdx::prelude::*;
#[derive(Debug, Clone, Copy)]
pub struct ConfigV2 {}

impl Params for ConfigV2 {
    type Vocab = Const<50280>;
    type DModel = Const<768>;
    type DtRank = Const<48>;
    type DConv = Const<4>;
    type DState = Const<16>;
    type DInner = Const<1536>;
    type DInnerX2 = Const<3072>;
    type XProjO = Const<80>;

    const MAX_SEQ_LEN: usize = 32768;
    const RMS_NORM_EPS: f64 = 1e-5;
    const LAYERS: usize = 24;

    fn vocab(&self) -> Self::Vocab {
        Self::Vocab {}
    }

    fn d_model(&self) -> Self::DModel {
        Self::DModel {}
    }

    fn dt_rank(&self) -> Self::DtRank {
        Self::DtRank {}
    }
    fn d_conv(&self) -> Self::DConv {
        Self::DConv {}
    }
    fn d_state(&self) -> Self::DState {
        Self::DState {}
    }
    fn d_inner(&self) -> Self::DInner {
        Self::DInner {}
    }

    fn d_inner_x2(&self) -> Self::DInnerX2 {
        Self::DInnerX2 {}
    }

    fn x_proj_o(&self) -> Self::XProjO {
        Self::XProjO {}
    }
}
