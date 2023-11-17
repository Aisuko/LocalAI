use burn::{
    config::Config,
    module::Module,
    nn::{self},
    tensor::{backend::Backend, Tensor},
};

use super::{MLPConfig, MultiHeadSelfAttention, MultiHeadSelfAttentionConfig, MLP};

#[derive(Config)]
pub struct ResidualEncoderAttentionBlockConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl ResidualEncoderAttentionBlockConfig {
    pub fn init<B: Backend>(&self) -> ResidualEncoderAttentionBlock<B> {
        let attn = MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init();
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init();

        let mlp = MLPConfig::new(self.n_state).init();
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init();

        ResidualEncoderAttentionBlock {
            attn,
            attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualEncoderAttentionBlock<B: Backend> {
    pub attn: MultiHeadSelfAttention<B>,
    pub attn_ln: nn::LayerNorm<B>,
    pub mlp: MLP<B>,
    pub mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualEncoderAttentionBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), None);
        let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        return x;
    }
}
