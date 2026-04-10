/// Input vocabulary: printable ASCII (32..=126) + PAD token
pub const IN_VOCAB_SIZE: usize = 96; // 95 printable ASCII + PAD
pub const IN_PAD: usize = 95;
pub const IN_MAX_LEN: usize = 128; // input context truncated to this

/// Output vocabulary: a-z, 0-9, hyphen, EOS
pub const OUT_VOCAB_SIZE: usize = 38;
pub const OUT_EOS: usize = 37;
pub const OUT_PAD: usize = 37; // same token — EOS also acts as pad
pub const OUT_MAX_LEN: usize = 32;

pub fn encode_input(s: &str) -> Vec<usize> {
    let mut ids: Vec<usize> = s
        .chars()
        .filter_map(|c| {
            let v = c as usize;
            if (32..=126).contains(&v) { Some(v - 32) } else { None }
        })
        .take(IN_MAX_LEN)
        .collect();
    // Pad to IN_MAX_LEN
    ids.resize(IN_MAX_LEN, IN_PAD);
    ids
}

pub fn encode_output(name: &str) -> Vec<usize> {
    let mut ids: Vec<usize> = name
        .chars()
        .filter_map(encode_out_char)
        .take(OUT_MAX_LEN - 1) // leave room for EOS
        .collect();
    ids.push(OUT_EOS);
    ids.resize(OUT_MAX_LEN, OUT_PAD);
    ids
}

pub fn decode_output(ids: &[usize]) -> String {
    ids.iter()
        .take_while(|&&id| id != OUT_EOS)
        .filter_map(|&id| decode_out_char(id))
        .collect()
}

fn encode_out_char(c: char) -> Option<usize> {
    match c {
        'a'..='z' => Some(c as usize - 'a' as usize),
        'A'..='Z' => Some(c as usize - 'A' as usize), // fold to lowercase
        '0'..='9' => Some(26 + (c as usize - '0' as usize)),
        '-' | '_' => Some(36),                          // both map to hyphen slot
        _ => None,
    }
}

fn decode_out_char(id: usize) -> Option<char> {
    match id {
        0..=25 => Some((b'a' + id as u8) as char),
        26..=35 => Some((b'0' + (id as u8 - 26)) as char),
        36 => Some('-'),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_output() {
        let name = "alice-profile";
        let encoded = encode_output(name);
        let decoded = decode_output(&encoded);
        assert_eq!(decoded, name);
    }

    #[test]
    fn encode_input_pads() {
        let v = encode_input("hi");
        assert_eq!(v.len(), IN_MAX_LEN);
        assert_eq!(v[2], IN_PAD);
    }
}
