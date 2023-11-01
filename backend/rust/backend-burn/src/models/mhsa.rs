//! Multi Head Self Attention
use crate::models::rope::RotaryEncoding;

use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{activation::softmax, backend::Backend, Tensor},
};

#[derive(Config)]
pub struct MultiHeadSelfAttentionConfig {
    n_state: usize,
    n_head: usize,
    n_kv_head: usize,
}

impl MultiHeadSelfAttentionConfig {
    /// Initializes a Multi-Head Self-Attention layer with the given configuration.
    pub fn init<B: Backend>(&self) -> MultiHeadSelfAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "n_state must be divisible by n_head"
        );
        assert!(
            self.n_state % self.n_kv_head == 0,
            "n_state must be divisible by n_kv_head"
        );

        let n_head_dim = self.n_state / self.n_head;
        let n_head = self.n_head;
        let n_kv_head = self.n_kv_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init();
        let key = nn::LinearConfig::new(self.n_state, n_kv_head * n_head_dim)
            .with_bias(false)
            .init();
        let value = nn::LinearConfig::new(self.n_state, n_kv_head * n_head_dim)
            .with_bias(false)
            .init();
        let out = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init();

        MultiHeadSelfAttention {
            n_head,
            n_kv_head,
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
    pub n_kv_head: usize,
    pub query: nn::Linear<B>,
    pub key: nn::Linear<B>,
    pub value: nn::Linear<B>,
    pub out: nn::Linear<B>,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    pub fn new(
        n_head: usize,
        n_kv_head: usize,
        query: nn::Linear<B>,
        key: nn::Linear<B>,
        value: nn::Linear<B>,
        out: nn::Linear<B>,
    ) -> Self {
        MultiHeadSelfAttention {
            n_head,
            n_kv_head,
            query,
            key,
            value,
            out,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rotary_encoder: &RotaryEncoding<B>,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        let wv = qkv_attention_rotary(q, k, v, mask, self.n_head, self.n_kv_head, rotary_encoder);
        return self.out.forward(wv);
    }
}

fn qkv_attention_rotary<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    mask: Option<Tensor<B, 2>>,
    n_head: usize,
    n_kv_head: usize,
    rotary_encoder: &RotaryEncoding<B>,
) -> Tensor<B, 3> {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_, n_ctx, _] = k.dims();
    let n_hstate = n_state / n_head;
    // keeps the value weightings roughly normally distributed
    let scale = (n_hstate as f64).powf(-0.25);
    let q = q.reshape([n_batch, n_qctx, n_head, n_hstate]);
    // interleave kv heads to match the number of q heads
    let n_repeat = n_head / n_kv_head;
    let k = repeat_kv(k.reshape([n_batch, n_ctx, n_kv_head, n_hstate]), n_repeat);
    let v = repeat_kv(v.reshape([n_batch, n_ctx, n_kv_head, n_hstate]), n_repeat);

    // the last two dims need to be (n_ctx, n_hstate)
    let q = rotary_encoder.forward(q.swap_dims(1, 2)) * scale;
    let k = rotary_encoder.forward(k.swap_dims(1, 2)) * scale;
    let v = v.swap_dims(1, 2);

    // compute value weightings
    let qk = q.matmul(k.transpose());

    // apply mask
    let qk = if let Some(mask) = mask {
        qk + mask.slice([0..n_qctx, 0..n_ctx]).unsqueeze::<4>()
    } else {
        qk
    };

    //normalize value weightings
    let w = softmax(qk, 3);
    let o = w.matmul(v).swap_dims(1, 2).flatten(2, 3);
    return o;
}

/// For a tensor size (n_batch, n_ctx, n_kv_head, n_hstate),
/// repeats the head keys or values in a interleaving manner so that the number
/// of heads is effectively multiplied by n_repeat.
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_repeat: usize) -> Tensor<B, 4> {
    if n_repeat > 1 {
        let [n_batch, n_ctx, n_kv_head, n_hstate] = x.dims();
        x.repeat(3, n_repeat)
            .reshape([n_batch, n_ctx, n_kv_head * n_repeat, n_hstate])
    } else {
        x
    }
}
