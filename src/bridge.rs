//! Wire format for the walker-delta ↔ LeoDOS bridge.
//!
//! Walker-delta publishes simulation state (orbital positions, LOS
//! bitmasks, time) to LeoDOS over UDP. LeoDOS-side hwlib backends
//! consume the same byte layout from a sibling module copy.
//!
//! Encoding rules:
//! - Big-endian (network byte order) for all multi-byte fields.
//! - `#[repr(C)]` + zerocopy for stable layout, no allocation,
//!   no serde overhead.
//! - One `StateHeader` followed by `num_sats × SatState` per UDP
//!   datagram. Receivers MUST validate `magic` and `version`.

use zerocopy::byteorder::network_endian::F64;
use zerocopy::byteorder::network_endian::U16;
use zerocopy::byteorder::network_endian::U32;
use zerocopy::byteorder::network_endian::U64;
use zerocopy::FromBytes;
use zerocopy::Immutable;
use zerocopy::IntoBytes;
use zerocopy::KnownLayout;
use zerocopy::Unaligned;

/// Magic bytes identifying a walker-delta → LeoDOS state packet.
pub const STATE_MAGIC: [u8; 4] = *b"LEOS";

/// Wire format version. Bump on any layout change.
pub const BRIDGE_VERSION: u16 = 1;

/// UDP port LeoDOS hwlib backends listen on for state.
pub const TOPOLOGY_PORT: u16 = 7000;

/// Direction enum encoded in `PacketEvent::link`.
pub const DIR_NORTH: u8 = 0;
pub const DIR_SOUTH: u8 = 1;
pub const DIR_EAST: u8 = 2;
pub const DIR_WEST: u8 = 3;
pub const DIR_GROUND: u8 = 4;

/// Header for a state packet. Followed by `num_sats × SatState`.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
pub struct StateHeader {
    pub magic: [u8; 4],
    pub version: U16,
    pub seq: U32,
    pub sim_time_ms: U64,
    pub real_time_ms: U64,
    pub num_sats: U16,
    pub _pad: [u8; 4],
}

impl StateHeader {
    pub fn new(seq: u32, sim_time_ms: u64, real_time_ms: u64, num_sats: u16) -> Self {
        Self {
            magic: STATE_MAGIC,
            version: U16::new(BRIDGE_VERSION),
            seq: U32::new(seq),
            sim_time_ms: U64::new(sim_time_ms),
            real_time_ms: U64::new(real_time_ms),
            num_sats: U16::new(num_sats),
            _pad: [0; 4],
        }
    }

    pub fn validate(&self) -> Result<(), DecodeError> {
        if self.magic != STATE_MAGIC {
            return Err(DecodeError::BadMagic);
        }
        if self.version.get() != BRIDGE_VERSION {
            return Err(DecodeError::VersionMismatch {
                expected: BRIDGE_VERSION,
                got: self.version.get(),
            });
        }
        Ok(())
    }
}

/// Per-satellite state. Position in ECI meters, velocity in m/s,
/// nadir attitude as a body→ECI quaternion (w, x, y, z).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
pub struct SatState {
    pub scid: U32,
    pub pos_eci_m: [F64; 3],
    pub vel_eci_m_s: [F64; 3],
    pub nadir_quat: [F64; 4],
    pub los_neighbors: u8,
    pub los_ground: U16,
    pub _pad: [u8; 9],
}

impl SatState {
    pub fn new(
        scid: u32,
        pos_eci_m: [f64; 3],
        vel_eci_m_s: [f64; 3],
        nadir_quat: [f64; 4],
        los_neighbors: u8,
        los_ground: u16,
    ) -> Self {
        Self {
            scid: U32::new(scid),
            pos_eci_m: [
                F64::new(pos_eci_m[0]),
                F64::new(pos_eci_m[1]),
                F64::new(pos_eci_m[2]),
            ],
            vel_eci_m_s: [
                F64::new(vel_eci_m_s[0]),
                F64::new(vel_eci_m_s[1]),
                F64::new(vel_eci_m_s[2]),
            ],
            nadir_quat: [
                F64::new(nadir_quat[0]),
                F64::new(nadir_quat[1]),
                F64::new(nadir_quat[2]),
                F64::new(nadir_quat[3]),
            ],
            los_neighbors,
            los_ground: U16::new(los_ground),
            _pad: [0; 9],
        }
    }

    pub fn los_has(&self, dir: u8) -> bool {
        (self.los_neighbors >> dir) & 1 == 1
    }
}

/// Errors decoding an incoming state packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Buffer too short for a header.
    HeaderTooShort,
    /// Header magic doesn't match.
    BadMagic,
    /// Wire format version unsupported.
    VersionMismatch { expected: u16, got: u16 },
    /// Buffer too short for the declared sat count.
    BodyTooShort { expected: usize, got: usize },
}

/// Encode a state packet into `out`. Returns the number of bytes
/// written. `out` must be at least
/// `size_of::<StateHeader>() + sats.len() * size_of::<SatState>()`.
pub fn encode_state(out: &mut [u8], header: &StateHeader, sats: &[SatState]) -> usize {
    let header_len = core::mem::size_of::<StateHeader>();
    let sat_len = core::mem::size_of::<SatState>();
    let total = header_len + sats.len() * sat_len;
    assert!(out.len() >= total, "buffer too small for state packet");
    out[..header_len].copy_from_slice(header.as_bytes());
    let mut off = header_len;
    for sat in sats {
        out[off..off + sat_len].copy_from_slice(sat.as_bytes());
        off += sat_len;
    }
    total
}

/// Decode a state packet. Returns header + view of the sat array.
pub fn decode_state(buf: &[u8]) -> Result<(StateHeader, &[SatState]), DecodeError> {
    let header_len = core::mem::size_of::<StateHeader>();
    if buf.len() < header_len {
        return Err(DecodeError::HeaderTooShort);
    }
    let header = StateHeader::read_from_bytes(&buf[..header_len])
        .map_err(|_| DecodeError::HeaderTooShort)?;
    header.validate()?;
    let n = header.num_sats.get() as usize;
    let body = &buf[header_len..];
    let expected = n * core::mem::size_of::<SatState>();
    if body.len() < expected {
        return Err(DecodeError::BodyTooShort {
            expected,
            got: body.len(),
        });
    }
    let sats = <[SatState]>::ref_from_bytes(&body[..expected])
        .map_err(|_| DecodeError::BodyTooShort {
            expected,
            got: body.len(),
        })?;
    Ok((header, sats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_is_stable() {
        assert_eq!(core::mem::size_of::<StateHeader>(), 32);
    }

    #[test]
    fn sat_state_size_is_stable() {
        assert_eq!(core::mem::size_of::<SatState>(), 96);
    }

    #[test]
    fn round_trip_empty() {
        let h = StateHeader::new(0, 0, 0, 0);
        let mut buf = [0u8; 32];
        let n = encode_state(&mut buf, &h, &[]);
        assert_eq!(n, 32);
        let (h2, sats) = decode_state(&buf[..n]).unwrap();
        assert_eq!(h2.magic, STATE_MAGIC);
        assert_eq!(h2.version.get(), BRIDGE_VERSION);
        assert_eq!(sats.len(), 0);
    }

    #[test]
    fn round_trip_three_sats() {
        let sats = [
            SatState::new(1, [7000e3, 0.0, 0.0], [0.0, 7.5e3, 0.0], [1.0, 0.0, 0.0, 0.0], 0b0011, 0),
            SatState::new(2, [0.0, 7000e3, 0.0], [-7.5e3, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0b1100, 0b0001),
            SatState::new(3, [0.0, 0.0, 7000e3], [0.0, 0.0, 7.5e3], [1.0, 0.0, 0.0, 0.0], 0b1111, 0b0011),
        ];
        let h = StateHeader::new(42, 1_000_000, 2_000_000, sats.len() as u16);
        let mut buf = vec![0u8; 32 + sats.len() * 96];
        let n = encode_state(&mut buf, &h, &sats);
        assert_eq!(n, buf.len());

        let (h2, decoded) = decode_state(&buf[..n]).unwrap();
        assert_eq!(h2.seq.get(), 42);
        assert_eq!(h2.sim_time_ms.get(), 1_000_000);
        assert_eq!(h2.real_time_ms.get(), 2_000_000);
        assert_eq!(h2.num_sats.get(), 3);
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].scid.get(), 1);
        assert_eq!(decoded[0].pos_eci_m[0].get(), 7000e3);
        assert_eq!(decoded[1].scid.get(), 2);
        assert_eq!(decoded[2].los_neighbors, 0b1111);
        assert_eq!(decoded[2].los_ground.get(), 0b0011);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = [0u8; 32];
        let h = StateHeader::new(0, 0, 0, 0);
        encode_state(&mut buf, &h, &[]);
        buf[0] = b'X';
        assert!(matches!(decode_state(&buf), Err(DecodeError::BadMagic)));
    }

    #[test]
    fn rejects_short_header() {
        let buf = [0u8; 8];
        assert!(matches!(decode_state(&buf), Err(DecodeError::HeaderTooShort)));
    }

    #[test]
    fn rejects_truncated_body() {
        let h = StateHeader::new(0, 0, 0, 2);
        let mut buf = vec![0u8; 32 + 96];
        encode_state(&mut buf[..32 + 96], &h, &[SatState::new(1, [0.0; 3], [0.0; 3], [1.0, 0.0, 0.0, 0.0], 0, 0)]);
        let truncated = &buf[..32 + 96];
        match decode_state(truncated) {
            Err(DecodeError::BodyTooShort { expected: 192, .. }) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn los_has_decodes_bitmask() {
        let s = SatState::new(0, [0.0; 3], [0.0; 3], [1.0, 0.0, 0.0, 0.0], 0b1010, 0);
        assert!(!s.los_has(DIR_NORTH));
        assert!(s.los_has(DIR_SOUTH));
        assert!(!s.los_has(DIR_EAST));
        assert!(s.los_has(DIR_WEST));
    }
}
