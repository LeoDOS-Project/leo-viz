//! TCP server for the leo-viz ↔ LeoDOS bridge.
//!
//! One server instance per launched constellation. Owns a
//! [`TcpListener`] on an ephemeral loopback port, accepts inbound
//! connections from cFS `sim_client` processes, reads a [`Hello`]
//! frame from each to learn the connecting satellite's SCID, and
//! sends one [`StateFrame`] per simulator tick to the matching
//! satellite.
//!
//! Thread model: a single accept thread receives new connections
//! and registers them in a shared map keyed by SCID. The main
//! (UI) thread calls [`BridgeServer::publish_tick`] on each frame;
//! that walks the map and writes one [`StateFrame`] per connected
//! client. Writes are blocking but the frames are small (128 B) and
//! the connections are loopback, so this is effectively a memcpy.

use crate::bridge::ENDPOINT_GROUND;
use crate::bridge::ENDPOINT_SATELLITE;
use crate::bridge::EventFrame;
use crate::bridge::GroundStateFrame;
use crate::bridge::Hello;
use crate::bridge::KIND_GROUND_STATE;
use crate::bridge::KIND_PING_REQUEST;
use crate::bridge::PingRequestFrame;
use crate::bridge::StateFrame;
use crate::bridge::VisibleSat;
use crate::walker::SatelliteState;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::Read;
use std::io::Write;
use std::net::SocketAddr;
use std::net::TcpListener;
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use zerocopy::FromBytes;
use zerocopy::IntoBytes;

const ACCEPT_POLL: Duration = Duration::from_millis(200);

/// One EVS event observed on a connected satellite, decoded from the
/// `EventFrame` wire format and stamped with a host-side receive time.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ReceivedEvent {
    pub scid: u32,
    pub sim_time_ms: u64,
    pub recv_real_time_ms: u64,
    pub event_id: u16,
    pub event_type: u8,
    pub app_name: String,
    pub message: String,
}

const MAX_EVENTS: usize = 4096;

#[allow(dead_code)]
pub struct BridgeServer {
    addr: SocketAddr,
    state: Arc<Mutex<ServerState>>,
    events: Arc<Mutex<VecDeque<ReceivedEvent>>>,
    stop: Arc<AtomicBool>,
    accept_thread: Option<JoinHandle<()>>,
    seq: u32,
}

pub struct ConnectionTracker {
    state: Arc<Mutex<ServerState>>,
}

impl ConnectionTracker {
    pub fn connected_scids(&self) -> Vec<u32> {
        self.state
            .lock()
            .map(|s| s.streams.keys().copied().collect())
            .unwrap_or_default()
    }
}

struct ServerState {
    /// Stream per connected satellite SCID. Insertion is from the
    /// accept thread; iteration + writes happen from the UI thread.
    streams: HashMap<u32, TcpStream>,
    /// Stream per connected ground-station id. Same threading rules
    /// as `streams`.
    ground_streams: HashMap<u32, TcpStream>,
}

impl BridgeServer {
    /// Bind a fresh ephemeral port on `127.0.0.1`. Returns an
    /// error if the host is out of ports.
    pub fn bind() -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let state = Arc::new(Mutex::new(ServerState {
            streams: HashMap::new(),
            ground_streams: HashMap::new(),
        }));
        let events = Arc::new(Mutex::new(VecDeque::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let accept_thread = {
            let state = Arc::clone(&state);
            let events = Arc::clone(&events);
            let stop = Arc::clone(&stop);
            thread::Builder::new()
                .name(format!("bridge-accept-{}", addr.port()))
                .spawn(move || run_accept(listener, state, events, stop))?
        };

        log::info!("bridge server bound to {}", addr);
        Ok(Self {
            addr,
            state,
            events,
            stop,
            accept_thread: Some(accept_thread),
            seq: 0,
        })
    }

    /// Drain all received events accumulated since the last call.
    #[allow(dead_code)]
    pub fn drain_events(&mut self) -> Vec<ReceivedEvent> {
        match self.events.lock() {
            Ok(mut g) => g.drain(..).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// A `Send`-able handle to the event queue for read-only access.
    #[allow(dead_code)]
    pub fn events_handle(&self) -> Arc<Mutex<VecDeque<ReceivedEvent>>> {
        Arc::clone(&self.events)
    }

    /// The address clients should connect to.
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Returns the SCIDs of all currently-connected satellite clients.
    pub fn connected_scids(&self) -> Vec<u32> {
        self.state
            .lock()
            .map(|s| s.streams.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Returns the station ids of all currently-connected ground daemons.
    pub fn connected_ground_ids(&self) -> Vec<u32> {
        self.state
            .lock()
            .map(|s| s.ground_streams.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Push a [`GroundStateFrame`] to one connected ground daemon.
    /// `visible` should already be ordered by elevation (highest first).
    /// Returns `true` if the frame was written; `false` if no daemon
    /// is connected for that station_id or the write failed (in which
    /// case the connection is dropped).
    pub fn publish_ground_tick(
        &mut self,
        station_id: u32,
        sim_time_seconds: f64,
        visible: &[VisibleSat],
    ) -> bool {
        let sim_time_ms = (sim_time_seconds * 1000.0) as u64;
        let frame = GroundStateFrame::new(self.seq, sim_time_ms, station_id, visible);
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        let Some(stream) = state.ground_streams.get_mut(&station_id) else {
            return false;
        };
        if write_kinded(stream, KIND_GROUND_STATE, frame.as_bytes()) {
            true
        } else {
            log::warn!("bridge server: dropping ground id={} (write failed)", station_id);
            state.ground_streams.remove(&station_id);
            false
        }
    }

    /// Send a [`PingRequestFrame`] to a connected ground daemon.
    /// Returns `false` if no daemon is connected for that id or
    /// the write failed.
    pub fn send_ping_request(&mut self, station_id: u32, frame: &PingRequestFrame) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        let Some(stream) = state.ground_streams.get_mut(&station_id) else {
            return false;
        };
        if write_kinded(stream, KIND_PING_REQUEST, frame.as_bytes()) {
            true
        } else {
            log::warn!("bridge server: dropping ground id={} (write failed)", station_id);
            state.ground_streams.remove(&station_id);
            false
        }
    }

    /// A `Send`-able read-only handle to the connection set; used by
    /// the cFS launcher thread to gate `docker exec`s on prior sats
    /// having actually connected.
    pub fn tracker(&self) -> ConnectionTracker {
        ConnectionTracker {
            state: Arc::clone(&self.state),
        }
    }

    /// Send one [`StateFrame`] per connected client, indexing into
    /// `sats` and `sats_next` by SCID. Slices are expected to be
    /// in plane-major SCID order, i.e. `sats[scid]` is the entry
    /// for that satellite. Drops connections whose write fails.
    pub fn publish_tick(
        &mut self,
        sim_time_seconds: f64,
        sats: &[SatelliteState],
        sats_next: &[SatelliteState],
        dt: f64,
    ) {
        let sim_time_ms = (sim_time_seconds * 1000.0) as u64;
        let real_time_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let Ok(mut state) = self.state.lock() else {
            return;
        };

        let mut to_drop = Vec::new();
        for (&scid, stream) in state.streams.iter_mut() {
            let i = scid as usize;
            let Some(s) = sats.get(i) else { continue };
            let s_next = sats_next.get(i).unwrap_or(s);
            let frame = build_frame(self.seq, sim_time_ms, real_time_ms, scid, s, s_next, dt);
            if stream.write_all(frame.as_bytes()).is_err() {
                to_drop.push(scid);
            }
        }
        for scid in to_drop {
            log::warn!("bridge server: dropping disconnected scid={}", scid);
            state.streams.remove(&scid);
        }
        self.seq = self.seq.wrapping_add(1);
    }
}

impl Drop for BridgeServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.accept_thread.take() {
            let _ = h.join();
        }
    }
}

/// Write a 1-byte kind tag followed by `body` as a single contiguous
/// buffer (one syscall, no Nagle interleaving with concurrent writers
/// to the same stream).
fn write_kinded(stream: &mut TcpStream, kind: u8, body: &[u8]) -> bool {
    let mut buf = Vec::with_capacity(1 + body.len());
    buf.push(kind);
    buf.extend_from_slice(body);
    stream.write_all(&buf).is_ok()
}

fn build_frame(
    seq: u32,
    sim_time_ms: u64,
    real_time_ms: u64,
    scid: u32,
    s: &SatelliteState,
    s_next: &SatelliteState,
    dt: f64,
) -> StateFrame {
    let pos_eci_m = [s.x * 1000.0, s.y * 1000.0, s.z * 1000.0];
    let vel_eci_m_s = [
        (s_next.x - s.x) * 1000.0 / dt,
        (s_next.y - s.y) * 1000.0 / dt,
        (s_next.z - s.z) * 1000.0 / dt,
    ];
    let nadir_quat = [1.0, 0.0, 0.0, 0.0];
    let mut los_neighbors: u8 = 0;
    for (i, _) in s.neighbors.iter().take(4).enumerate() {
        los_neighbors |= 1 << i;
    }
    StateFrame::new(
        seq,
        sim_time_ms,
        real_time_ms,
        scid,
        pos_eci_m,
        vel_eci_m_s,
        nadir_quat,
        los_neighbors,
        0,
    )
}

fn run_accept(
    listener: TcpListener,
    state: Arc<Mutex<ServerState>>,
    events: Arc<Mutex<VecDeque<ReceivedEvent>>>,
    stop: Arc<AtomicBool>,
) {
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _peer)) => {
                let state = Arc::clone(&state);
                let events = Arc::clone(&events);
                thread::Builder::new()
                    .name("bridge-handshake".into())
                    .spawn(move || handle_new_connection(stream, state, events))
                    .ok();
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(ACCEPT_POLL);
            }
            Err(e) => {
                log::warn!("bridge server accept failed: {}", e);
                thread::sleep(ACCEPT_POLL);
            }
        }
    }
}

fn handle_new_connection(
    mut stream: TcpStream,
    state: Arc<Mutex<ServerState>>,
    events: Arc<Mutex<VecDeque<ReceivedEvent>>>,
) {
    if let Err(e) = stream.set_nonblocking(false) {
        log::warn!("bridge server: failed to set blocking: {}", e);
        return;
    }
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));

    let mut buf = [0u8; core::mem::size_of::<Hello>()];
    if let Err(e) = stream.read_exact(&mut buf) {
        log::warn!("bridge server: hello read failed: {}", e);
        return;
    }
    let Ok(hello) = Hello::read_from_bytes(&buf) else {
        log::warn!("bridge server: hello decode failed");
        return;
    };
    if let Err(e) = hello.validate() {
        log::warn!("bridge server: hello rejected: {:?}", e);
        return;
    }
    let id = hello.scid.get();
    let kind = hello.endpoint_kind;

    let _ = stream.set_read_timeout(None);

    // try_clone gives us a separate file descriptor referring to the
    // same TCP connection; the read-side thread can own it
    // independently while the UI thread keeps the original for writes.
    let read_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            log::warn!("bridge server: try_clone failed for id={}: {}", id, e);
            return;
        }
    };

    if let Ok(mut s) = state.lock() {
        let map = match kind {
            ENDPOINT_GROUND => &mut s.ground_streams,
            // Treat anything we don't know as a satellite for backward compat.
            _ => &mut s.streams,
        };
        let label = match kind {
            ENDPOINT_GROUND => "ground",
            ENDPOINT_SATELLITE => "sat",
            _ => "sat",
        };
        if let Some(old) = map.insert(id, stream) {
            log::warn!("bridge server: replacing existing {} connection for id={}", label, id);
            drop(old);
        } else {
            log::info!("bridge server: {} id={} connected", label, id);
        }
    }

    thread::Builder::new()
        .name(format!("bridge-events-{}", id))
        .spawn(move || event_reader(read_stream, id, events))
        .ok();
}

fn event_reader(
    mut stream: TcpStream,
    scid: u32,
    events: Arc<Mutex<VecDeque<ReceivedEvent>>>,
) {
    let mut buf = [0u8; core::mem::size_of::<EventFrame>()];
    loop {
        if stream.read_exact(&mut buf).is_err() {
            log::info!("bridge server: event reader for scid={} ended", scid);
            return;
        }
        let Ok(frame) = EventFrame::read_from_bytes(&buf) else {
            log::warn!("bridge server: event decode failed for scid={}", scid);
            continue;
        };
        if frame.validate().is_err() {
            log::warn!("bridge server: event validation failed for scid={}", scid);
            continue;
        }
        let recv_real_time_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let received = ReceivedEvent {
            scid: frame.scid.get(),
            sim_time_ms: frame.sim_time_ms.get(),
            recv_real_time_ms,
            event_id: frame.event_id.get(),
            event_type: frame.event_type,
            app_name: frame.app_name_str().to_string(),
            message: frame.message_str().to_string(),
        };
        if let Ok(mut q) = events.lock() {
            q.push_back(received);
            while q.len() > MAX_EVENTS {
                q.pop_front();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::BRIDGE_MAGIC;
    use crate::bridge::BRIDGE_VERSION;

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
            neighbors: vec![1, 2],
            name: None,
            tle_inclination_deg: None,
            tle_mean_motion: None,
        }
    }

    #[test]
    fn handshake_then_one_frame() {
        let mut server = BridgeServer::bind().unwrap();
        let addr = server.local_addr();

        // Client side: connect, send Hello{scid=1}, read one StateFrame.
        let client_thread = thread::spawn(move || {
            let mut s = TcpStream::connect(addr).unwrap();
            let hello = Hello::new(1);
            s.write_all(hello.as_bytes()).unwrap();
            let mut buf = [0u8; core::mem::size_of::<StateFrame>()];
            s.read_exact(&mut buf).unwrap();
            let frame = StateFrame::read_from_bytes(&buf).unwrap();
            (frame.scid.get(), frame.pos_eci_m[0].get())
        });

        // Wait for accept thread to register the new connection.
        for _ in 0..50 {
            if !server.connected_scids().is_empty() {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }

        let sats = vec![fake_sat(0, 6000.0), fake_sat(1, 7000.0)];
        let sats_next = vec![fake_sat(0, 6000.5), fake_sat(1, 7000.5)];
        server.publish_tick(0.0, &sats, &sats_next, 1.0);

        let (scid, x_m) = client_thread.join().unwrap();
        assert_eq!(scid, 1);
        assert_eq!(x_m, 7_000_000.0);
        // Sanity: magic+version were validated on the way out.
        let _ = (BRIDGE_MAGIC, BRIDGE_VERSION);
    }
}
