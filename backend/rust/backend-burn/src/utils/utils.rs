//! utils function for the models

use std::f32::NEG_INFINITY;

use burn::tensor::{backend::Backend, Tensor};

/// Computes the power of a given base to each element of a tensor.
///
/// # Arguments
///
/// * `base` - A floating-point number representing the base.
/// * `x` - A tensor of any dimensionality.
///
/// # Returns
///
/// A tensor of the same dimensionality as `x`, where each element is the result of raising `base` to the power of the corresponding element in `x`.
pub fn powto<B: Backend, const D: usize>(base: f64, x: Tensor<B, D>) -> Tensor<B, D> {
    let baseln = base.ln();
    x.mul_scalar(baseln).exp()
}

/// Generates a strictly upper traingular matrix filled with -inf that when added to an attention weight matrix prevents vectors
/// from attending to other vectors further in the sequence, preventing future information from flowing into the past.
pub fn attn_decoder_mask<B: Backend>(seq_length: usize) -> Tensor<B, 2> {
    let mut mask = Tensor::<B, 2>::zeros([seq_length, seq_length]);

    for i in 0..(seq_length - 1) {
        let values = Tensor::<B, 2>::zeros([1, seq_length - (i + 1)]).add_scalar(NEG_INFINITY);
        mask = mask.slice_assign([i..i + 1, i + 1..seq_length], values);
    }
    return mask;
}
