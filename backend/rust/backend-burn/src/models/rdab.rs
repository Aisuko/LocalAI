//! Residual Decoder Attention

use crate::models::mhsa::MultiHeadSelfAttention;
use crate::models::mlp::MLP;
use crate::models::rmsnorm::RMSNorm;

use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Tensor},
};

use crate::models::{MLPConfig, MultiHeadSelfAttentionConfig, RMSNormConfig, RotaryEncoding};

#[derive(Config)]
pub struct ResidualDecoderAttentionBlockConfig {
    n_state: usize,
    multiple_of: usize,
    ffn_dim_multiplier: Option<usize>,
    n_head: usize,
    n_kv_head: usize,
    norm_eps: f64,
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    pub attn: MultiHeadSelfAttention<B>,
    pub attn_norm: RMSNorm<B>,
    pub mlp: MLP<B>,
    pub mlp_norm: RMSNorm<B>,
}

impl ResidualDecoderAttentionBlockConfig {
    /// Initializes a new residual decoder attention block with the given configuration.
    pub fn init<B: Backend>(&self) -> ResidualDecoderAttentionBlock<B> {
        let attn =
            MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head, self.n_kv_head).init();
        let attn_norm = RMSNormConfig::new(self.n_state, self.norm_eps).init();

        let mlp = MLPConfig::new(self.n_state, 4 * self.n_state, self.multiple_of)
            .with_ffn_dim_multiplier(self.ffn_dim_multiplier)
            .init();

        let mlp_norm = RMSNormConfig::new(self.n_state, self.norm_eps).init();

        ResidualDecoderAttentionBlock {
            attn,
            attn_norm,
            mlp,
            mlp_norm,
        }
    }
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    pub fn new(
        attn: MultiHeadSelfAttention<B>,
        attn_norm: RMSNorm<B>,
        mlp: MLP<B>,
        mlp_norm: RMSNorm<B>,
    ) -> Self {
        ResidualDecoderAttentionBlock {
            attn,
            attn_norm,
            mlp,
            mlp_norm,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rotary_encoder: &RotaryEncoding<B>,
        mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let x = x.clone()
            + self
                .attn
                .forward(self.attn_norm.forward(x), rotary_encoder, Some(mask));
        let x = x.clone() + self.mlp.forward(self.mlp_norm.forward(x));
        return x;
    }
}
