#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Spec §3.A0 (Header).
pub struct Header {
    pub ver: u8,
    pub typ: u8,
    pub length: u8,
}

impl Header {
    pub fn to_bytes(self) -> [u8; 2] {
        let b0 = ((self.ver & 0x0f) << 4) | (self.typ & 0x0f);
        [b0, self.length]
    }

    pub fn from_bytes(b: [u8; 2]) -> Self {
        let ver = (b[0] >> 4) & 0x0f;
        let typ = b[0] & 0x0f;
        let length = b[1];
        Self { ver, typ, length }
    }
}

const CRC32C_POLY_REFLECTED: u32 = 0x82F6_3B78;

const fn crc32c_table() -> [u32; 256] {
    let mut tbl = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            if (c & 1) != 0 {
                c = (c >> 1) ^ CRC32C_POLY_REFLECTED;
            } else {
                c >>= 1;
            }
            k += 1;
        }
        tbl[i] = c;
        i += 1;
    }
    tbl
}

const CRC32C_TBL: [u32; 256] = crc32c_table();

pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xffff_ffff;
    for &b in data {
        let idx = ((crc ^ (b as u32)) & 0xff) as usize;
        crc = CRC32C_TBL[idx] ^ (crc >> 8);
    }
    crc ^ 0xffff_ffff
}

fn bytes_to_bits_be(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 8);
    for &b in bytes {
        for bit in (0..8).rev() {
            out.push(((b >> bit) & 1) as u8);
        }
    }
    out
}

fn bits_be_to_bytes(bits: &[u8]) -> Vec<u8> {
    assert!(bits.len() % 8 == 0);
    let mut out = Vec::with_capacity(bits.len() / 8);
    for chunk in bits.chunks_exact(8) {
        let mut b = 0u8;
        for &v in chunk {
            b = (b << 1) | (v & 1);
        }
        out.push(b);
    }
    out
}

/// Spec §3.A0.
pub fn build_u_bits(payload: &[u8], ver: u8, typ: u8) -> anyhow::Result<Vec<u8>> {
    if payload.len() > 26 {
        anyhow::bail!("payload too long for K=256 (max 26 bytes)");
    }
    let hdr = Header {
        ver: ver & 0x0f,
        typ: typ & 0x0f,
        length: payload.len() as u8,
    };
    let hdr_b = hdr.to_bytes();
    let mut msg = Vec::with_capacity(2 + payload.len() + 4);
    msg.extend_from_slice(&hdr_b);
    msg.extend_from_slice(payload);
    let crc = crc32c(&msg);
    msg.extend_from_slice(&crc.to_be_bytes());

    while msg.len() < 32 {
        msg.push(0);
    }
    let bits = bytes_to_bits_be(&msg);
    debug_assert_eq!(bits.len(), 256);
    Ok(bits)
}

pub fn parse_u_bits(u_bits: &[u8]) -> anyhow::Result<(Header, Vec<u8>, bool)> {
    if u_bits.len() != 256 {
        anyhow::bail!("U must be 256 bits");
    }
    let bytes = bits_be_to_bytes(u_bits);
    let hdr = Header::from_bytes([bytes[0], bytes[1]]);
    if hdr.length > 26 {
        return Ok((hdr, Vec::new(), false));
    }
    let end_payload = 2usize + (hdr.length as usize);
    let end_crc = end_payload + 4;
    if end_crc > bytes.len() {
        return Ok((hdr, Vec::new(), false));
    }
    let payload = bytes[2..end_payload].to_vec();
    let crc_rx = u32::from_be_bytes(bytes[end_payload..end_crc].try_into().unwrap());
    let crc_ok = crc32c(&bytes[..end_payload]) == crc_rx;
    Ok((hdr, payload, crc_ok))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_matches_python_vector() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn u_bits_roundtrip() -> anyhow::Result<()> {
        let payload = b"hello";
        let u = build_u_bits(payload, 1, 1)?;
        let (hdr, pl, ok) = parse_u_bits(&u)?;
        assert!(ok);
        assert_eq!(hdr.ver, 1);
        assert_eq!(hdr.typ, 1);
        assert_eq!(hdr.length as usize, payload.len());
        assert_eq!(pl, payload);
        Ok(())
    }
}
