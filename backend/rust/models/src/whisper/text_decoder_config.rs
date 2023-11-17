use burn::{
    config::Config,
    module::{Module, Param},
    nn::{self},
    tensor::{backend::Backend, module::embedding, Distribution, Int, Tensor},
};

use super::{
    attn_decoder_mask, ResidualDecoderAttentionBlock, ResidualDecoderAttentionBlockConfig,
};

#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
}

impl TextDecoderConfig {
    pub fn init<B: Backend>(&self) -> TextDecoder<B> {
        let token_embedding = Tensor::random(
            [self.n_vocab, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
        )
        .into();
        let positional_embedding = Tensor::random(
            [self.n_text_ctx, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
        )
        .into();
        let blocks: Vec<_> = (0..self.n_text_layer)
            .into_iter()
            .map(|_| {
                ResidualDecoderAttentionBlockConfig::new(self.n_text_state, self.n_text_head).init()
            })
            .collect();
        let ln = nn::LayerNormConfig::new(self.n_text_state).init();
        let mask = attn_decoder_mask(self.n_text_ctx).into();

        let n_vocab = self.n_vocab;
        let n_text_ctx = self.n_text_ctx;

        TextDecoder {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            n_vocab,
            n_text_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    pub token_embedding: Param<Tensor<B, 2>>,
    pub positional_embedding: Param<Tensor<B, 2>>,
    pub blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    pub ln: nn::LayerNorm<B>,
    pub mask: Param<Tensor<B, 2>>,
    pub n_vocab: usize,
    pub n_text_ctx: usize,
}

impl<B: Backend> TextDecoder<B> {
    pub fn forward(&self, x: Tensor<B, 2, Int>, xa: Tensor<B, 3>) -> Tensor<B, 3> {
        let [n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.n_text_ctx,
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.n_text_ctx
        );

        let x = embedding(self.token_embedding.val(), x)
            + self
                .positional_embedding
                .val()
                .slice([0..seq_len])
                .unsqueeze::<3>();

        let mut x = x;

        for block in &self.blocks {
            x = block.forward(x, xa.clone(), self.mask.val());
        }

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, xa.clone(), self.mask.val());
        }

        let x = self.ln.forward(x);
        return x.matmul(self.token_embedding.val().transpose().unsqueeze::<3>());
    }

    pub fn ctx_size(&self) -> usize {
        self.n_text_ctx
    }
}
