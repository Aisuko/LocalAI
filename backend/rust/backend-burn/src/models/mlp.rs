//! The role of Multi Layer Perceptron in the Llama2 architecture is to transform the input embeddings
//! into a higher dimensional space, which is then passed thorugh the attention mechanism in the encoder block.
use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{backend::Backend, Tensor},
};

use crate::models::silu::SILU;

#[derive(Config)]
pub struct MLPConfig {
    n_state: usize,
    n_state_hidden: usize,
    multiple_of: usize,
    ffn_dim_multiplier: Option<usize>,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self) -> MLP<B> {
        let mut hidden_dim = 2 * self.n_state_hidden / 3;
        if let Some(ffn_dim_multiplier) = self.ffn_dim_multiplier {
            hidden_dim = ffn_dim_multiplier * hidden_dim;
        }
        hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) / self.multiple_of);

        let w1 = nn::LinearConfig::new(self.n_state, hidden_dim)
            .with_bias(false)
            .init();
        let w2 = nn::LinearConfig::new(hidden_dim, self.n_state)
            .with_bias(false)
            .init();
        let w3 = nn::LinearConfig::new(self.n_state, hidden_dim)
            .with_bias(false)
            .init();

        let silu = SILU::new();

        MLP { w1, w2, w3, silu }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub w1: nn::Linear<B>,
    pub w2: nn::Linear<B>,
    pub w3: nn::Linear<B>,
    pub silu: SILU,
}

impl<B: Backend> MLP<B> {
    pub fn new(w1: nn::Linear<B>, w2: nn::Linear<B>, w3: nn::Linear<B>, silu: SILU) -> Self {
        MLP { w1, w2, w3, silu }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.w2
            .forward(self.silu.forward(self.w1.forward(x.clone())) * self.w3.forward(x))
    }
}
