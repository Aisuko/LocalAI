use std::f32::NEG_INFINITY;

use burn::{
    config::Config,
    module::Module,
    nn::{self},
    tensor::{backend::Backend, Int, Tensor},
};

use super::{AudioEncoder, AudioEncoderConfig, TextDecoder, TextDecoderConfig};

#[derive(Config, Debug)]
pub struct WhisperConfig {
    pub audio_encoder_config: AudioEncoderConfig,
    pub text_decoder_config: TextDecoderConfig,
}

impl WhisperConfig {
    pub fn init<B: Backend>(&self) -> Whisper<B> {
        let n_audio_state = self.audio_encoder_config.n_audio_state;
        let n_text_state = self.text_decoder_config.n_text_state;

        assert!(
            n_audio_state == n_text_state,
            "Audio encoder state size {} must be equal to text decoder state size {}.",
            n_audio_state,
            n_text_state
        );

        let encoder = self.audio_encoder_config.init();
        let decoder = self.text_decoder_config.init();

        Whisper { encoder, decoder }
    }
}

#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    pub encoder: AudioEncoder<B>,
    pub decoder: TextDecoder<B>,
}

impl<B: Backend> Whisper<B> {
    pub fn forward(&self, mel: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.decoder.forward(tokens, self.encoder.forward(mel))
    }

    pub fn forward_encoder(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel)
    }

    pub fn forward_decoder(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.decoder.forward(tokens, encoder_output)
    }

    pub fn encoder_ctx_size(&self) -> usize {
        self.encoder.ctx_size()
    }

    pub fn decoder_ctx_size(&self) -> usize {
        self.decoder.ctx_size()
    }
}
