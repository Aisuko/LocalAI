pub(crate) mod mnist;
pub(crate) mod whisper;

pub use mnist::mnist::MNINST;

use bunker::pb::{ModelOptions, PredictOptions};
/// Trait for implementing a Language Model.
pub trait LLM {
    /// Creates a new instance of the Language Model.
    fn new(request: ModelOptions) -> Self;
    /// Loads the model from the given options.
    fn load_model(&mut self, request: ModelOptions) -> Result<String, Box<dyn std::error::Error>>;
    /// Predicts the output for the given input options.
    fn predict(&mut self, request: PredictOptions) -> Result<String, Box<dyn std::error::Error>>;
}
