//! Sigmoid Linear Unit (SiLU) activation function used in Neural Networks.
//! The activation of SILU is computed by the sigmoid function multiplied by its input,
//! or x*sigmoid(x).

use burn::{
    module::Module,
    tensor::{activation::sigmoid, backend::Backend, Tensor},
};

#[derive(Module, Clone, Debug)]
pub struct SILU {}

impl SILU {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        x.clone() * sigmoid(x)
    }
}
