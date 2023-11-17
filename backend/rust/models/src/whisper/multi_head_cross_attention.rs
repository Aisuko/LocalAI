use burn::{
    config::Config,
    module::Module,
    nn::{self},
    tensor::{backend::Backend, Tensor},
};

use super::qkv_attention;

#[derive(Config)]
pub struct MultiHeadCrossAttentionConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl MultiHeadCrossAttentionConfig {
    pub fn init<B: Backend>(&self) -> MultiHeadCrossAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "State size {} must be a multiple of head size {}",
            self.n_state,
            self.n_head
        );

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init();
        let key = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init();
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init();
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init();

        MultiHeadCrossAttention {
            n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadCrossAttention<B: Backend> {
    pub n_head: usize,
    pub query: nn::Linear<B>,
    pub key: nn::Linear<B>,
    pub value: nn::Linear<B>,
    pub out: nn::Linear<B>,
}

impl<B: Backend> MultiHeadCrossAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, xa: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.query.forward(x);
        let k = self.key.forward(xa.clone());
        let v = self.value.forward(xa);

        let wv = qkv_attention(q, k, v, None, self.n_head);

        return self.out.forward(wv);
    }
}
