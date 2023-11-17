use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        self,
        conv::{Conv1d, Conv1dConfig},
        PaddingConfig1d,
    },
    tensor::{backend::Backend, Distribution, Tensor},
};

use super::{ResidualEncoderAttentionBlock, ResidualEncoderAttentionBlockConfig};

#[derive(Config, Debug)]
pub struct AudioEncoderConfig {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
}

impl AudioEncoderConfig {
    pub fn init<B: Backend>(&self) -> AudioEncoder<B> {
        let conv1 = Conv1dConfig::new(self.n_mels, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init();
        let gelu1 = nn::GELU::new();
        let conv2 = Conv1dConfig::new(self.n_audio_state, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init();
        let gelu2 = nn::GELU::new();

        let blocks: Vec<_> = (0..self.n_audio_layer)
            .into_iter()
            .map(|_| {
                ResidualEncoderAttentionBlockConfig::new(self.n_audio_state, self.n_audio_head)
                    .init()
            })
            .collect();

        let ln_post = nn::LayerNormConfig::new(self.n_audio_state).init();
        let positional_embedding = Tensor::random(
            [self.n_audio_ctx, self.n_audio_state],
            Distribution::Normal(0.0, 1.0),
        )
        .into();

        let n_mels = self.n_mels;
        let n_audio_ctx = self.n_audio_ctx;

        AudioEncoder {
            conv1,
            gelu1,
            conv2,
            gelu2,
            blocks,
            ln_post,
            positional_embedding,
            n_mels,
            n_audio_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub gelu1: nn::GELU,
    pub conv2: Conv1d<B>,
    pub gelu2: nn::GELU,
    pub blocks: Vec<ResidualEncoderAttentionBlock<B>>,
    pub ln_post: nn::LayerNorm<B>,
    pub positional_embedding: Param<Tensor<B, 2>>,
    pub n_mels: usize,
    pub n_audio_ctx: usize,
}

impl<B: Backend> AudioEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, n_mels, n_ctx] = x.dims();

        assert!(
            n_mels == self.n_mels,
            "Audio mel spectrum size must be {}.",
            self.n_mels
        );

        assert!(
            n_ctx <= self.n_audio_ctx,
            "Audio length {} cannot exceed {}.",
            n_ctx,
            self.n_audio_ctx
        );

        let x = self.gelu1.forward(self.conv1.forward(x));
        let x = self.gelu2.forward(self.conv2.forward(x));

        let x = x.swap_dims(1, 2);
        let k = x.dims()[1];

        let x = x + self
            .positional_embedding
            .val()
            .slice([0..k])
            .unsqueeze::<3>();

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }

        return self.ln_post.forward(x);
    }

    pub fn ctx_size(&self) -> usize {
        self.n_audio_ctx
    }
}
