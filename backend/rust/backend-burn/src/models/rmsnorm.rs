//! Root Mean Square Layer Normalization https://dl.acm.org/doi/pdf/10.5555/3454287.3455397
//! The RMS Norm is a type of normalization used in the Llama2 architecture.
//! It is a pre-normalization technique that normalizes the input of each transformer sub-layer
//! using the Root Mean Square (RMS) method.
//!
//! It improves the performance of the model by normalizing the input of each transformer sub-layer.

use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Config)]
pub struct RMSNormConfig {
    layer_size: usize,
    eps: f64,
}

impl RMSNormConfig {
    pub fn init<B: Backend>(&self) -> RMSNorm<B> {
        assert!(self.eps > 0.0, "eps must be positive.");

        let weight = Tensor::ones([self.layer_size]);
        let eps = self.eps;
        RMSNorm { weight, eps }
    }
}

#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    pub weight: Tensor<B, 1>,
    pub eps: f64,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(weight: Tensor<B, 1>, eps: f64) -> Self {
        assert!(eps > 0.0, "eps must be positive.");

        RMSNorm { weight, eps }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let rms = (x.clone().powf(2.0).mean_dim(D - 1) + self.eps).sqrt();
        (x / rms) * self.weight.clone().unsqueeze()
    }
}
