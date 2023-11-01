//! Llama2 model and it's configuration

use crate::models::rdab::{ResidualDecoderAttentionBlock, ResidualDecoderAttentionBlockConfig};
use crate::models::rmsnorm::RMSNorm;
use crate::models::rope::RotaryEncoding;
use crate::utils::utils::attn_decoder_mask;

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{backend::Backend, Int, Tensor},
};

use super::{RMSNormConfig, RotaryEncodingConfig};

#[derive(Config, Debug)]
pub struct LlamaConfig {
    n_vocab: usize,
    n_ctx: usize,
    n_state: usize,
    multiple_of: usize,
    ffn_dim_multiplier: Option<usize>,
    n_head: usize,
    n_kv_head: usize,
    n_layer: usize,
    #[config(default = 1e-6)]
    norm_eps: f64,
}

#[derive(Module, Debug)]
pub struct Llama<B: Backend> {
    pub token_embedding: nn::Embedding<B>,
    pub rotary_encoder: RotaryEncoding<B>,
    pub blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    pub norm: RMSNorm<B>,
    pub output: nn::Linear<B>,
    pub mask: Param<Tensor<B, 2>>,
    pub n_vocab: usize,
    pub n_ctx: usize,
}

impl LlamaConfig {
    pub fn init<B: Backend>(&self) -> Llama<B> {
        let token_embedding = nn::EmbeddingConfig::new(self.n_vocab, self.n_state).init();
        let rotary_encoder =
            RotaryEncodingConfig::new(self.n_ctx, self.n_state / self.n_head, 10000.0).init();
        let blocks: Vec<_> = (0..self.n_layer)
            .into_iter()
            .map(|_| {
                ResidualDecoderAttentionBlockConfig::new(
                    self.n_state,
                    self.multiple_of,
                    self.n_head,
                    self.n_kv_head,
                    self.norm_eps,
                )
                .with_ffn_dim_multiplier(self.ffn_dim_multiplier)
                .init()
            })
            .collect();

        let norm = RMSNormConfig::new(self.n_state, self.norm_eps).init();
        let output = nn::LinearConfig::new(self.n_state, self.n_vocab)
            .with_bias(false)
            .init();

        let mask = attn_decoder_mask(self.n_ctx).into();

        let n_vocab = self.n_vocab;
        let n_ctx = self.n_ctx;

        Llama {
            token_embedding,
            rotary_encoder,
            blocks,
            norm,
            output,
            mask,
            n_vocab,
            n_ctx,
        }
    }
}

impl<B: Backend> Llama<B> {
    pub fn new(
        token_embedding: nn::Embedding<B>,
        rotary_encoder: RotaryEncoding<B>,
        blocks: Vec<ResidualDecoderAttentionBlock<B>>,
        norm: RMSNorm<B>,
        output: nn::Linear<B>,
        mask: Param<Tensor<B, 2>>,
        n_vocab: usize,
        n_ctx: usize,
    ) -> Self {
        Self {
            token_embedding,
            rotary_encoder,
            blocks,
            norm,
            output,
            mask,
            n_vocab,
            n_ctx,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.n_ctx,
            "Sequence length {} is greater than the maximum sequence length {}",
            seq_len,
            self.n_ctx
        );

        let x = self.token_embedding.forward(x);
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, &self.rotary_encoder, self.mask.val());
        }

        self.output.forward(self.norm.forward(x))
    }
}
