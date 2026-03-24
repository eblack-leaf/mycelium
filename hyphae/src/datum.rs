pub struct Datum {
    nl: String,
    intent: usize,
    entity: usize,
    projection_fields: Vec<usize>,
    condition_fields: Vec<usize>,
    condition_comparators: Vec<usize>,
    modifier_types: Vec<usize>,
    modifier_fields: Vec<usize>,
    modifier_directions: Vec<bool>,
    assignment_fields: Vec<usize>,
    assignment_value_types: Vec<usize>,
}
impl Datum {

}