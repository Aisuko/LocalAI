use burn::{
    config::Config,
    module::Module,
    nn::{self},
    tensor::{backend::Backend, Tensor},
};

use super::{
    MLPConfig, MultiHeadCrossAttention, MultiHeadCrossAttentionConfig, MultiHeadSelfAttention,
    MultiHeadSelfAttentionConfig, MLP,
};

#[derive(Config)]
pub struct ResidualDecoderAttentionBlockConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl ResidualDecoderAttentionBlockConfig {
    pub fn init<B: Backend>(&self) -> ResidualDecoderAttentionBlock<B> {
        let attn = MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init();
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init();

        let cross_attn = MultiHeadCrossAttentionConfig::new(self.n_state, self.n_head).init();
        let cross_attn_ln = nn::LayerNormConfig::new(self.n_state).init();

        let mlp = MLPConfig::new(self.n_state).init();
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init();

        ResidualDecoderAttentionBlock {
            attn,
            attn_ln,
            cross_attn,
            cross_attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    pub attn: MultiHeadSelfAttention<B>,
    pub attn_ln: nn::LayerNorm<B>,
    pub cross_attn: MultiHeadCrossAttention<B>,
    pub cross_attn_ln: nn::LayerNorm<B>,
    pub mlp: MLP<B>,
    pub mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>, xa: Tensor<B, 3>, mask: Tensor<B, 2>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(x, Some(mask));
        let x = x.clone() + self.cross_attn.forward(self.cross_attn_ln.forward(x), xa);
        let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        return x;
    }
}
