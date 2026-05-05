//! Wire format for the leo-viz ↔ LeoDOS bridge over TCP.
//!
//! `leo-viz` runs a TCP server per launched constellation. Each cFS
//! `sim_client` opens an outbound connection at boot and:
//!   1. writes one [`Hello`] identifying its spacecraft id;
//!   2. reads a stream of [`StateFrame`]s, one per simulator tick,
//!      each carrying the satellite's current ECI position/velocity,
//!      attitude, and link visibility.
//!
//! Encoding rules:
//! - Big-endian (network byte order) for all multi-byte fields.
//! - `#[repr(C)]` + zerocopy for stable layout, no allocation,
//!   no serde overhead.
//! - Both message types are fixed size — no length prefix needed.
//!   Receivers MUST validate `magic` and `version` on every frame
//!   to recover from a stream gone out of sync.

use zerocopy::byteorder::network_endian::F64;
use zerocopy::byteorder::network_endian::U16;
use zerocopy::byteorder::network_endian::U32;
use zerocopy::byteorder::network_endian::U64;
use zerocopy::FromBytes;
use zerocopy::Immutable;
use zerocopy::IntoBytes;
use zerocopy::KnownLayout;
use zerocopy::Unaligned;

/// Magic bytes identifying any frame in the leo-viz ↔ LeoDOS protocol.
pub const BRIDGE_MAGIC: [u8; 4] = *b"LEOS";

/// Wire format version. Bump on any layout change.
pub const BRIDGE_VERSION: u16 = 2;

/// Default loopback port. leo-viz allocates one ephemeral listener
/// per launched constellation, so this is only a fallback for
/// standalone tests; production launches pass an explicit port via
/// the `LEODOS_BRIDGE_ADDR` env var.
pub const DEFAULT_BRIDGE_PORT: u16 = 7000;

/// North direction in `los_neighbors` bitmask.
pub const DIR_NORTH: u8 = 0;
/// South direction in `los_neighbors` bitmask.
pub const DIR_SOUTH: u8 = 1;
/// East direction in `los_neighbors` bitmask.
pub const DIR_EAST: u8 = 2;
/// West direction in `los_neighbors` bitmask.
pub const DIR_WEST: u8 = 3;
/// Ground link encoded separately from torus neighbors.
pub const DIR_GROUND: u8 = 4;

/// Client → server: identifies which satellite this connection is.
/// Sent once immediately after the TCP handshake, before any
/// [`StateFrame`]s are read.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
pub struct Hello {
    pub magic: [u8; 4],
    pub version: U16,
    pub _pad0: [u8; 2],
    pub scid: U32,
    pub _pad1: [u8; 4],
}

impl Hello {
    pub fn new(scid: u32) -> Self {
        Self {
            magic: BRIDGE_MAGIC,
            version: U16::new(BRIDGE_VERSION),
            _pad0: [0; 2],
            scid: U32::new(scid),
            _pad1: [0; 4],
        }
    }

    pub fn validate(&self) -> Result<(), DecodeError> {
        if self.magic != BRIDGE_MAGIC {
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

/// Server → client: per-tick snapshot of one satellite's state.
/// Position in ECI meters, velocity in m/s, nadir attitude as a
/// body→ECI quaternion (w, x, y, z).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
pub struct StateFrame {
    pub magic: [u8; 4],
    pub version: U16,
    pub _pad0: [u8; 2],
    /// Monotonic sequence number assigned by leo-viz.
    pub seq: U32,
    /// Simulated mission time in milliseconds since epoch.
    pub sim_time_ms: U64,
    /// Wall clock time in milliseconds when the frame was published.
    pub real_time_ms: U64,
    /// Spacecraft id this frame is addressed to (matches `Hello.scid`).
    pub scid: U32,
    pub _pad1: [u8; 4],
    /// ECI position in meters.
    pub pos_eci_m: [F64; 3],
    /// ECI velocity in m/s.
    pub vel_eci_m_s: [F64; 3],
    /// Body→ECI quaternion (w, x, y, z) for nadir-pointing attitude.
    pub nadir_quat: [F64; 4],
    /// Bitmask of torus neighbors currently in line of sight.
    pub los_neighbors: u8,
    pub _pad2: [u8; 1],
    /// Bitmask of ground stations currently in view.
    pub los_ground: U16,
    pub _pad3: [u8; 4],
}

impl StateFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        seq: u32,
        sim_time_ms: u64,
        real_time_ms: u64,
        scid: u32,
        pos_eci_m: [f64; 3],
        vel_eci_m_s: [f64; 3],
        nadir_quat: [f64; 4],
        los_neighbors: u8,
        los_ground: u16,
    ) -> Self {
        Self {
            magic: BRIDGE_MAGIC,
            version: U16::new(BRIDGE_VERSION),
            _pad0: [0; 2],
            seq: U32::new(seq),
            sim_time_ms: U64::new(sim_time_ms),
            real_time_ms: U64::new(real_time_ms),
            scid: U32::new(scid),
            _pad1: [0; 4],
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
            _pad2: [0; 1],
            los_ground: U16::new(los_ground),
            _pad3: [0; 4],
        }
    }

    pub fn validate(&self) -> Result<(), DecodeError> {
        if self.magic != BRIDGE_MAGIC {
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

    /// Returns `true` if `dir` (one of `DIR_*`) is in the neighbor mask.
    pub fn los_has(&self, dir: u8) -> bool {
        (self.los_neighbors >> dir) & 1 == 1
    }
}

/// Errors decoding an incoming frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Buffer was too short for the expected frame size.
    Truncated { expected: usize, got: usize },
    /// Frame magic did not match [`BRIDGE_MAGIC`].
    BadMagic,
    /// Wire format version did not match [`BRIDGE_VERSION`].
    VersionMismatch { expected: u16, got: u16 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_size_is_stable() {
        assert_eq!(core::mem::size_of::<Hello>(), 16);
    }

    #[test]
    fn state_frame_size_is_stable() {
        assert_eq!(core::mem::size_of::<StateFrame>(), 124);
    }

    #[test]
    fn hello_round_trip() {
        let h = Hello::new(42);
        let bytes = h.as_bytes();
        let decoded = Hello::read_from_bytes(bytes).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded.scid.get(), 42);
    }

    #[test]
    fn state_frame_round_trip() {
        let f = StateFrame::new(
            7,
            1_000_000,
            2_000_000,
            3,
            [7000e3, 0.0, 0.0],
            [0.0, 7.5e3, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            0b0011,
            0b0001,
        );
        let bytes = f.as_bytes();
        let decoded = StateFrame::read_from_bytes(bytes).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded.seq.get(), 7);
        assert_eq!(decoded.scid.get(), 3);
        assert_eq!(decoded.pos_eci_m[0].get(), 7000e3);
        assert!(decoded.los_has(DIR_NORTH));
        assert!(decoded.los_has(DIR_SOUTH));
        assert!(!decoded.los_has(DIR_EAST));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut h = Hello::new(1);
        h.magic = *b"XXXX";
        assert!(matches!(h.validate(), Err(DecodeError::BadMagic)));
    }
}
