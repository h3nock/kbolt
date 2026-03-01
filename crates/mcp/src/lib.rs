use kbolt_core::engine::Engine;

pub struct McpAdapter {
    pub engine: Engine,
}

impl McpAdapter {
    pub fn new(engine: Engine) -> Self {
        Self { engine }
    }
}
