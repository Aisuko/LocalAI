use burn::{
    config::Config,
    module::Module,
    nn::{self},
    tensor::{backend::Backend, Tensor},
};

use super::qkv_attention;

#[derive(Config)]
pub struct MultiHeadSelfAttentionConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl MultiHeadSelfAttentionConfig {
    pub fn init<B: Backend>(&self) -> MultiHeadSelfAttention<B> {
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

        MultiHeadSelfAttention {
            n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    pub n_head: usize,
    pub query: nn::Linear<B>,
    pub key: nn::Linear<B>,
    pub value: nn::Linear<B>,
    pub out: nn::Linear<B>,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        let wv = qkv_attention(q, k, v, None, self.n_head);

        return self.out.forward(wv);
    }
}
