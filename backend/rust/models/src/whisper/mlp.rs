use burn::{
    config::Config,
    module::Module,
    nn::{self},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config)]
pub struct MLPConfig {
    pub n_state: usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self) -> MLP<B> {
        let lin1 = nn::LinearConfig::new(self.n_state, self.n_state * 4).init();
        let gelu = nn::GELU::new();
        let lin2 = nn::LinearConfig::new(self.n_state * 4, self.n_state).init();

        MLP { lin1, gelu, lin2 }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub lin1: nn::Linear<B>,
    pub gelu: nn::GELU,
    pub lin2: nn::Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.lin1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.lin2.forward(x);
        return x;
    }
}
