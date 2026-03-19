// stipe — main interface, connects UI to the network

pub struct Prompt {}

pub struct Query {}

pub struct Mycelium {}

impl Mycelium {
    pub fn new() -> Self {
        Self {}
    }

    pub fn query(&self, _prompt: Prompt) -> Query {
        todo!()
    }
}
