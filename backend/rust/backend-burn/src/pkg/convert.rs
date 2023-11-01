use crate::models::{Llama, LlamaConfig};

use burn::{
    // backend::tch::{TchBackend, TchDevice},
    config::Config,
    module::Module,
    tensor::backend::Backend,
};

use burn::record::{BinFileRecorder, HalfPrecisionSettings, Recorder, RecorderError};

use crate::pkg::loader::Loader;

pub struct Convertion {}

impl Convertion {
    pub fn convert_llama_dump_to_model<B: Backend>(
        dump_path: &str,
        model_name: &str,
        device: &B::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (llama, llama_conifg): (Llama<B>, LlamaConfig) =
            Loader::load_llama_dmp(dump_path, device)?;


        Convertion::save_llama_model_file(llama, model_name)?;
        llama_conifg.save(&format!("{model_name}.cfg"))?;
        Ok(())
    }

    pub fn save_llama_model_file<B: Backend>(
        llama: Llama<B>,
        name: &str,
    ) -> Result<(), RecorderError> {
        BinFileRecorder::<HalfPrecisionSettings>::new().record(llama.into_record(), name.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convertion_ndarray() {
        use burn::backend::ndarray::{NdArrayBackend, NdArrayDevice};
        type Backend = NdArrayBackend<f32>;
        let device = NdArrayDevice::Cpu;

        let home = std::env::var("HOME").unwrap();

        let dump_path = &format!("{}/Downloads/workspace/kimchi/params", home);
        let model_name = &"llama2-7b-chat".to_string();

        let option =
            Convertion::convert_llama_dump_to_model::<Backend>(dump_path, model_name, &device);

        assert!(option.is_ok());
    }
}
