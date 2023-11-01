//! LLM tokenization tools crate.
//! Adapted from the https://github.com/Gadersd/llama2-burn/blob/main/src/token.rs
//! Adapted by Aisuko

use rust_tokenizers::{
    error::TokenizerError,
    tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy},
    vocab::Vocab,
};

use std::{result, vec};

const BOS_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;

pub type Result<T> = result::Result<T, TokenizerError>;

pub struct LlamaTokenizer {
    spm: SentencePieceBpeTokenizer,
}

impl LlamaTokenizer {
    pub fn new(tokenizer_path: &str) -> Result<Self> {
        let lower_case = false;
        SentencePieceBpeTokenizer::from_file(tokenizer_path, lower_case).map(|spm| Self { spm })
    }

    pub fn encode(&self, text: &str, inlcude_bos: bool, include_eos: bool) -> Vec<i64> {
        let pre = if inlcude_bos {
            vec![BOS_TOKEN_ID]
        } else {
            vec![]
        };

        let post = if include_eos {
            vec![EOS_TOKEN_ID]
        } else {
            vec![]
        };

        let token_ids = self
            .spm
            .encode(
                text,
                None,
                std::usize::MAX,
                &TruncationStrategy::LongestFirst,
                0,
            )
            .token_ids;

        [pre, token_ids, post]
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect()
    }

    pub fn decode(&self, tokens: &[i64], skip_special_tokens: bool) -> String {
        let clean_spaces = false;
        self.spm.decode(tokens, skip_special_tokens, clean_spaces)
    }

    pub fn vocab_size(&self, include_special_tokens: bool) -> usize {
        let vocab = self.spm.vocab();
        if include_special_tokens {
            vocab.values().len() + vocab.special_values().len()
        } else {
            vocab.values().len()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let home = std::env::var("HOME").unwrap();
        let tm_path = &format!("{}/Downloads/workspace/llama/tokenizer.model", home);
        let tokenizer = LlamaTokenizer::new(tm_path).unwrap();
        // tokenizer.vocab_size(fale) should be >0
        assert!(tokenizer.vocab_size(false) > 0);

        let test_str = "Hello, I am Llama2!";
        let encoded = tokenizer.encode(test_str, true, true);
        let decoded = tokenizer.decode(&encoded, false);

        assert!(encoded.len() > 0);
        // decided should contain test_str
        assert!(decoded.contains(test_str));
    }
}
