//! Loading all the module to the memory.
//! Adapted by Aisuko

use core::f32;
use num_traits::cast::ToPrimitive;

use burn::{
    nn,
    nn::{EmbeddingConfig, EmbeddingRecord},
    tensor::{backend::Backend, Data, ElementConversion, Tensor},
};

use crate::models::{
    Llama, LlamaConfig, MultiHeadSelfAttention, RMSNorm, ResidualDecoderAttentionBlock,
    RotaryEncodingConfig, MLP, SILU,
};
use crate::utils::utils::attn_decoder_mask;

pub struct Loader {}

impl Loader {
    pub fn numpy_to_tensor<B: Backend, const D: usize>(
        numpy_data: Vec<f32>,
        device: &B::Device,
    ) -> Tensor<B, D> {
        let v: Vec<f32> = numpy_data.to_vec();
        let shape: Vec<_> = v[0..D].into_iter().map(|&v| v as usize).collect();
        let data: Vec<B::FloatElem> = v[D..].into_iter().map(|e| e.elem()).collect();

        Tensor::from_data_device(Data::new(data, shape.into()), device)
    }

    pub fn load_tensor<B: Backend, const D: usize>(
        name: &str,
        path: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, D>, Box<dyn std::error::Error>> {
        let tensor_path = format!("{}/{}.npy", path, name);

        // TODO: check npyz behavior same to npy
        let bytes = std::fs::read(tensor_path)?;
        let reader = npyz::NpyFile::new(&bytes[..])?;
        let vec = reader.into_vec::<f32>()?;

        let tensor = Loader::numpy_to_tensor(vec, device);
        Ok(tensor)
    }

    pub fn load_f32<B: Backend>(
        name: &str,
        path: &str,
        device: &B::Device,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        Loader::load_tensor::<B, 1>(name, path, device).map(|t| t.into_scalar().to_f32().unwrap())
    }

    pub fn load_usize<B: Backend>(
        name: &str,
        path: &str,
        device: &B::Device,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        Loader::load_tensor::<B, 1>(name, path, device).map(|t| t.into_scalar().to_usize().unwrap())
    }

    pub fn load_linear<B: Backend>(
        path: &str,
        device: &B::Device,
    ) -> Result<nn::Linear<B>, Box<dyn std::error::Error>> {
        let weight = Loader::load_tensor::<B, 2>("weight", path, device)?;
        let bias = Loader::load_tensor::<B, 1>("bias", path, device).ok();

        let record = nn::LinearRecord {
            weight: weight.into(),
            bias: bias.map(|t| t.into()),
        };

        let linear: nn::Linear<B> = nn::LinearConfig::new(3, 3).init_with(record);
        Ok(linear)
    }

    pub fn load_rmsnorm<B: Backend>(
        path: &str,
        device: &B::Device,
    ) -> Result<RMSNorm<B>, Box<dyn std::error::Error>> {
        let weight = Loader::load_tensor::<B, 1>("weight", path, device)?;
        let eps = Loader::load_f32::<B>("eps", path, device)?.into();

        let rmsnorm = RMSNorm::new(weight.into(), eps);
        Ok(rmsnorm)
    }

    pub fn load_attention<B: Backend>(
        path: &str,
        device: &B::Device,
    ) -> Result<MultiHeadSelfAttention<B>, Box<dyn std::error::Error>> {
        let query = Loader::load_linear(&format!("{}/{}", path, "wq"), device)?;
        let key = Loader::load_linear(&format!("{}/{}", path, "wk"), device)?;
        let value = Loader::load_linear(&format!("{}/{}", path, "wv"), device)?;
        let out = Loader::load_linear(&format!("{}/{}", path, "wo"), device)?;

        let n_head = Loader::load_usize::<B>("n_head", path, device)?;
        let n_kv_head = Loader::load_usize::<B>("n_kv_head", path, device)?;

        let attention = MultiHeadSelfAttention::new(n_head, n_kv_head, query, key, value, out);

        Ok(attention)
    }

    pub fn load_feedforward<B: Backend>(
        path: &str,
        device: &B::Device,
    ) -> Result<MLP<B>, Box<dyn std::error::Error>> {
        let w1 = Loader::load_linear(&format!("{}/{}", path, "w1"), device)?;
        let w2 = Loader::load_linear(&format!("{}/{}", path, "w2"), device)?;
        let w3 = Loader::load_linear(&format!("{}/{}", path, "w3"), device)?;

        let mlp = MLP::new(w1, w2, w3, SILU::new());
        Ok(mlp)
    }

    pub fn load_transformer_block<B: Backend>(
        path: &str,
        device: &B::Device,
    ) -> Result<ResidualDecoderAttentionBlock<B>, Box<dyn std::error::Error>> {
        let attn = Loader::load_attention(&format!("{}/{}", path, "attention"), device)?;
        let attn_norm = Loader::load_rmsnorm(&format!("{}/{}", path, "attention_norm"), device)?;
        let mlp = Loader::load_feedforward(&format!("{}/{}", path, "feedforward"), device)?;
        let mlp_norm = Loader::load_rmsnorm(&format!("{}/{}", path, "ffn_norm"), device)?;

        let block = ResidualDecoderAttentionBlock::new(attn, attn_norm, mlp, mlp_norm);

        Ok(block)
    }

    pub fn load_llama_dmp<B: Backend>(
        path: &str,
        device: &B::Device,
    ) -> Result<(Llama<B>, LlamaConfig), Box<dyn std::error::Error>> {
        let mut blocks: Vec<ResidualDecoderAttentionBlock<B>> = vec![];
        let n_layer = Loader::load_usize::<B>("n_layer", path, device)?;
        for i in 0..n_layer {
            let block = Loader::load_transformer_block(&format!("{}/layer{}", path, i), device)?;
            blocks.push(block);
        }

        let n_ctx = Loader::load_usize::<B>("n_ctx", path, device)?;
        let theta = Loader::load_f32::<B>("theta", path, device)?;
        let multiple_of = Loader::load_usize::<B>("multiple_of", path, device)?;
        let ffn_dim_multiplier = Loader::load_usize::<B>("ffn_dim_multiplier", path, device).ok();

        // We do not exactly know the dimension of the token_embedding
        let token_embedding = Loader::load_tensor("tok_embeddings/weight", path, device)?;
        let [n_vocab, n_state] = token_embedding.dims();
        let n_head = blocks[0].attn.n_head;
        let n_kv_head = blocks[0].attn.n_kv_head;
        let head_dim = n_state / n_head;

        let token_embedding = EmbeddingConfig::new(n_vocab, n_state).init_with(EmbeddingRecord {
            weight: token_embedding.into(),
        });

        let rotary_encoding = RotaryEncodingConfig::new(n_ctx, head_dim, theta.into()).init();

        let norm = Loader::load_rmsnorm(&format!("{}/{}", path, "norm"), device)?;
        let output = Loader::load_linear(&format!("{}/{}", path, "output"), device)?;
        let mask = attn_decoder_mask(n_ctx).into();
        let norm_eps = norm.eps;

        let llama = Llama::new(
            token_embedding,
            rotary_encoding,
            blocks,
            norm,
            output,
            mask,
            n_vocab,
            n_ctx,
        );

        let llama_config = LlamaConfig::new(
            n_vocab,
            n_ctx,
            n_state,
            multiple_of,
            n_head,
            n_kv_head,
            n_layer,
        )
        .with_norm_eps(norm_eps)
        .with_ffn_dim_multiplier(ffn_dim_multiplier);
        Ok((llama, llama_config))
    }
}
