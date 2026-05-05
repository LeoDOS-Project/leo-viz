//! UDP publisher for the walker-delta ↔ LeoDOS bridge.
//!
//! Opt-in via the `LEODOS_BRIDGE_ADDR` env var (e.g.
//! `127.0.0.1:7000`). When unset, the publisher is `None` and
//! the simulator runs unchanged.
//!
//! The publisher snapshots the active constellation each tick
//! (rate-limited by [`MIN_PUBLISH_INTERVAL`]) and sends one UDP
//! datagram per snapshot using the wire format defined in
//! [`crate::bridge`].

use crate::bridge::SatState;
use crate::bridge::StateHeader;
use crate::bridge::encode_state;
use crate::walker::SatelliteState;
use std::net::SocketAddr;
use std::net::UdpSocket;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

const ENV_VAR: &str = "LEODOS_BRIDGE_ADDR";
const MIN_PUBLISH_INTERVAL: Duration = Duration::from_millis(100);
const MAX_DATAGRAM_BYTES: usize = 65_000;

pub struct BridgePublisher {
    socket: UdpSocket,
    target: SocketAddr,
    seq: u32,
    last_publish: Option<Instant>,
    buf: Vec<u8>,
}

impl BridgePublisher {
    /// Open the publisher if `LEODOS_BRIDGE_ADDR` is set. Returns
    /// `None` when the env var is absent or invalid.
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var(ENV_VAR).ok()?;
        let target = raw.parse::<SocketAddr>().ok()?;
        let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
        socket.set_nonblocking(true).ok()?;
        log::info!("bridge publisher → {} (UDP)", target);
        Some(Self {
            socket,
            target,
            seq: 0,
            last_publish: None,
            buf: Vec::with_capacity(MAX_DATAGRAM_BYTES),
        })
    }

    /// Publish a snapshot. Rate-limited; returns immediately if
    /// the previous publish was more recent than [`MIN_PUBLISH_INTERVAL`].
    /// `sim_time_seconds` is the simulator's current sim clock.
    pub fn publish(&mut self, sim_time_seconds: f64, sats: &[SatelliteState]) {
        let now = Instant::now();
        if let Some(last) = self.last_publish {
            if now.duration_since(last) < MIN_PUBLISH_INTERVAL {
                return;
            }
        }

        let n = sats.len().min(u16::MAX as usize);
        let total = core::mem::size_of::<StateHeader>() + n * core::mem::size_of::<SatState>();
        if total > MAX_DATAGRAM_BYTES {
            log::warn!("bridge publisher: {} sats exceeds datagram budget", n);
            return;
        }

        let sim_time_ms = (sim_time_seconds * 1000.0) as u64;
        let real_time_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let header = StateHeader::new(self.seq, sim_time_ms, real_time_ms, n as u16);
        self.buf.resize(total, 0);

        let encoded: Vec<SatState> = sats.iter().take(n).map(encode_sat).collect();
        encode_state(&mut self.buf, &header, &encoded);

        match self.socket.send_to(&self.buf, self.target) {
            Ok(_) => {
                self.seq = self.seq.wrapping_add(1);
                self.last_publish = Some(now);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => log::warn!("bridge publisher send failed: {}", e),
        }
    }
}

#[cfg(test)]
impl BridgePublisher {
    fn new_to(target: SocketAddr) -> std::io::Result<Self> {
        let socket = UdpSocket::bind("127.0.0.1:0")?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            target,
            seq: 0,
            last_publish: None,
            buf: Vec::with_capacity(MAX_DATAGRAM_BYTES),
        })
    }
}

fn encode_sat(s: &SatelliteState) -> SatState {
    let pos_eci_m = [s.x * 1000.0, s.y * 1000.0, s.z * 1000.0];
    let vel_eci_m_s = [0.0, 0.0, 0.0];
    let nadir_quat = [1.0, 0.0, 0.0, 0.0];
    let mut los_neighbors: u8 = 0;
    for (i, _) in s.neighbors.iter().take(4).enumerate() {
        los_neighbors |= 1 << i;
    }
    SatState::new(
        s.sat_index as u32,
        pos_eci_m,
        vel_eci_m_s,
        nadir_quat,
        los_neighbors,
        0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::decode_state;

    fn fake_sat(idx: usize, x_km: f64) -> SatelliteState {
        SatelliteState {
            plane: 0,
            sat_index: idx,
            x: x_km,
            y: 0.0,
            z: 0.0,
            lat: 0.0,
            lon: 0.0,
            ascending: true,
            neighbors: vec![1, 2, 3],
            name: None,
            tle_inclination_deg: None,
            tle_mean_motion: None,
        }
    }

    #[test]
    fn publishes_decodable_packet() {
        let listener = UdpSocket::bind("127.0.0.1:0").unwrap();
        listener.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
        let target = listener.local_addr().unwrap();

        let mut pub_ = BridgePublisher::new_to(target).unwrap();
        let sats = vec![fake_sat(7, 7000.0), fake_sat(8, 7100.0)];
        pub_.publish(123.456, &sats);

        let mut buf = [0u8; 65_000];
        let (n, _) = listener.recv_from(&mut buf).unwrap();
        let (header, decoded) = decode_state(&buf[..n]).unwrap();

        assert_eq!(header.seq.get(), 0);
        assert_eq!(header.sim_time_ms.get(), 123_456);
        assert_eq!(header.num_sats.get(), 2);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].scid.get(), 7);
        assert_eq!(decoded[0].pos_eci_m[0].get(), 7_000_000.0);
        assert_eq!(decoded[0].los_neighbors, 0b0111);
        assert_eq!(decoded[1].scid.get(), 8);
    }

    #[test]
    fn rate_limits_publishes() {
        let listener = UdpSocket::bind("127.0.0.1:0").unwrap();
        listener.set_read_timeout(Some(Duration::from_millis(200))).unwrap();
        let target = listener.local_addr().unwrap();

        let mut pub_ = BridgePublisher::new_to(target).unwrap();
        let sats = vec![fake_sat(1, 7000.0)];
        pub_.publish(0.0, &sats);
        pub_.publish(0.001, &sats);

        let mut buf = [0u8; 1024];
        let first = listener.recv_from(&mut buf);
        assert!(first.is_ok());
        let second = listener.recv_from(&mut buf);
        assert!(second.is_err(), "second publish should have been rate-limited");
    }
}
