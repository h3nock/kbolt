use kbolt_core::engine::Engine;

#[derive(Debug)]
pub struct CliAdapter {
    pub engine: Engine,
}

impl CliAdapter {
    pub fn new(engine: Engine) -> Self {
        Self { engine }
    }
}
