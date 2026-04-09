use std::sync::Mutex;

pub(crate) type DataM = Mutex<Data>;
pub(crate) struct Data {

}
impl Data {
    pub fn new() -> Self {
        Self {}
    }
}