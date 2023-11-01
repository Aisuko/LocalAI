//! Rotary Positional Embedding (RoPE), which is a type of position embedding that encodes absolute positional information with a rotation matrix
//! and naturally incorporates decaying inter-token dependency with increasing relative distances
//! https://arxiv.org/abs/2104.09864
//! Adapted from https://github.com/Gadersd/llama2-burn/blob/main/src/model.rs#L404
//! Adapted by Aisuko

use crate::utils::utils::*;
use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct RotaryEncodingConfig {
    max_sequence_length: usize,
    state_size: usize,
    theta: f64,
}

impl RotaryEncodingConfig {
    pub fn init<B: Backend>(&self) -> RotaryEncoding<B> {
        //!TODO: This should be solved by a better way
        assert!(self.state_size % 2 == 0, "Head size must be even");
        assert!(self.theta > 0.0, "Theta must be positive");
        let half_state_size = self.state_size / 2;
        let arrange_m = Tensor::from_floats([[1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0]]).into();

        let inv_freq = powto(
            self.theta,
            Tensor::arange(0..half_state_size).float() * (2.0 / self.state_size as f64),
        )
        .powf(-1.0);

        let periods = Tensor::arange(0..self.max_sequence_length)
            .float()
            .unsqueeze::<2>()
            .transpose()
            .repeat(1, half_state_size)
            * inv_freq.unsqueeze();

        let p_cos = periods.clone().cos();
        let p_sin = periods.sin();
        let freq_cis = Tensor::cat(vec![p_cos, p_sin], 1)
            .reshape([self.max_sequence_length, 2, half_state_size])
            .transpose()
            .repeat(2, 2)
            .reshape([self.max_sequence_length, self.state_size, 2])
            .into();

        RotaryEncoding {
            arrange_m,
            freq_cis,
        }
    }
}

/// pairs the value of a vector (v0 v1 v2 ... vn) into complex numbers (c0 c1 c2 ... cn/2)
/// which are then roatetd counter-clockwise by the angle seq_index / theta^(2*pair_index/n).
/// This encodes sequence positions in a way that is agnostic to the maximum sequence length
/// which potentially allows for arbitrailly long sequences without retraining.
#[derive(Module, Debug)]
pub struct RotaryEncoding<B: Backend> {
    pub arrange_m: Param<Tensor<B, 2>>,
    pub freq_cis: Param<Tensor<B, 3>>,
}

impl<B: Backend> RotaryEncoding<B> {
    /// Applies rotary positional encoding to a tensor of dimenions (..., seq_len, n_state)
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor of shape (..., seq_len, n_state)
    ///
    /// # Returns
    ///
    /// The output tensor after applying rotary positional encoding of the same shape as the input tensor.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        assert!(D >= 2);
        let orig_shape = x.shape();
        let (n_ctx, n_state) = (orig_shape.dims[D - 2], orig_shape.dims[D - 1]);
        let dummy_dim_size = orig_shape.num_elements() / (n_ctx * n_state);

        let out = x
            .reshape([dummy_dim_size, n_ctx, n_state / 2, 2])
            .matmul(self.arrange_m.val().unsqueeze())
            .reshape([dummy_dim_size, n_ctx, n_state, 2])
            * self.freq_cis.val().slice([0..n_ctx]).unsqueeze();

        out.sum_dim(D - 1).reshape(orig_shape)
    }
}
