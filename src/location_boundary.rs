//! Selected-place geometry lookups; cached and serialized across Earth views.

use crate::aoi::preview::Preview;
use egui::Context;
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use web_time::Instant;

const DEFAULT_ENDPOINT: &str = "https://nominatim.openstreetmap.org/";
const CACHE_LIMIT: usize = 32;
const TIMEOUT: Duration = Duration::from_secs(25);
const INTERVAL: Duration = Duration::from_millis(1100);
const ENDPOINT_ID: &str = "location_boundary_endpoint";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OsmId {
    kind: char,
    id: u64,
}

impl OsmId {
    pub fn new(kind: &str, id: u64) -> Option<Self> {
        let kind = match kind {
            "N" | "node" => 'N',
            "W" | "way" => 'W',
            "R" | "relation" => 'R',
            _ => return None,
        };
        (id > 0).then_some(Self { kind, id })
    }
}

pub struct Area {
    pub preview: Arc<Preview>,
    pub extent: [f64; 4],
}

pub type Result = std::result::Result<Option<Arc<Area>>, String>;

#[derive(Clone)]
pub enum Status {
    Pending,
    Ready(Result),
}

pub type Ticket = Arc<Mutex<Status>>;

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    endpoint: String,
    osm: OsmId,
}

struct Job {
    key: Key,
    anchor: [f64; 2],
}

#[derive(Default)]
struct Service {
    entries: HashMap<Key, Ticket>,
    queue: VecDeque<Job>,
    completed: VecDeque<Key>,
    active: Option<(Key, Instant)>,
    last_start: Option<Instant>,
}

fn service() -> &'static Mutex<Service> {
    static SERVICE: OnceLock<Mutex<Service>> = OnceLock::new();
    SERVICE.get_or_init(Mutex::default)
}

pub fn endpoint(ctx: &Context) -> String {
    ctx.data_mut(|data| {
        let id = egui::Id::new(ENDPOINT_ID);
        data.get_persisted::<String>(id).unwrap_or_else(|| {
            #[cfg(not(target_arch = "wasm32"))]
            let configured = std::env::var("LEO_VIZ_NOMINATIM_URL").ok();
            #[cfg(target_arch = "wasm32")]
            let configured = web_sys::window()
                .and_then(|w| w.location().href().ok())
                .and_then(|href| url::Url::parse(&href).ok())
                .and_then(|url| {
                    url.query_pairs()
                        .find(|(key, _)| key == "nominatim_url")
                        .map(|(_, value)| value.into_owned())
                });
            let endpoint = configured.unwrap_or_else(|| DEFAULT_ENDPOINT.into());
            data.insert_persisted(id, endpoint.clone());
            endpoint
        })
    })
}

pub fn settings(ui: &mut egui::Ui) {
    let mut endpoint = endpoint(ui.ctx());
    ui.label("Endpoint");
    if ui
        .add(
            egui::TextEdit::singleline(&mut endpoint)
                .desired_width(ui.available_width().min(360.0)),
        )
        .changed()
    {
        ui.data_mut(|data| data.insert_persisted(egui::Id::new(ENDPOINT_ID), endpoint));
    }
    ui.hyperlink_to(
        "Usage policy",
        "https://operations.osmfoundation.org/policies/nominatim/",
    );
}

pub fn request(ctx: &Context, osm: OsmId, anchor: [f64; 2]) -> Ticket {
    let key = Key {
        endpoint: endpoint(ctx),
        osm,
    };
    let ticket = service().lock().unwrap().request(key, anchor);
    ctx.request_repaint();
    ticket
}

impl Service {
    fn request(&mut self, key: Key, anchor: [f64; 2]) -> Ticket {
        if let Some(ticket) = self.entries.get(&key) {
            if !matches!(&*ticket.lock().unwrap(), Status::Ready(Err(_))) {
                return Arc::clone(ticket);
            }
            self.entries.remove(&key);
            self.completed.retain(|old| old != &key);
        }
        let ticket = Arc::new(Mutex::new(Status::Pending));
        self.entries.insert(key.clone(), Arc::clone(&ticket));
        self.queue.push_back(Job { key, anchor });
        ticket
    }

    fn next(&mut self, now: Instant) -> Option<Job> {
        if let Some((key, started)) = &self.active {
            if now.duration_since(*started) < TIMEOUT {
                return None;
            }
            let key = key.clone();
            finish(
                self,
                &key,
                Err("Boundary lookup timed out. Please try again.".into()),
            );
        }
        if self
            .last_start
            .is_some_and(|last| now.duration_since(last) < INTERVAL)
        {
            return None;
        }
        loop {
            let job = self.queue.pop_front()?;
            if self
                .entries
                .get(&job.key)
                .is_some_and(|ticket| Arc::strong_count(ticket) > 1)
            {
                self.last_start = Some(now);
                self.active = Some((job.key.clone(), now));
                return Some(job);
            }
            self.entries.remove(&job.key);
        }
    }
}

pub fn cached(ctx: &Context, osm: OsmId) -> Option<Arc<Area>> {
    let key = Key {
        endpoint: endpoint(ctx),
        osm,
    };
    let service = service().lock().unwrap();
    let ticket = service.entries.get(&key)?;
    let area = match &*ticket.lock().unwrap() {
        Status::Ready(Ok(Some(area))) => Some(Arc::clone(area)),
        _ => None,
    };
    area
}

fn lookup_url(key: &Key) -> std::result::Result<String, String> {
    let base = format!("{}/", key.endpoint.trim_end_matches('/'));
    let url = url::Url::parse(&base).map_err(|_| "Invalid boundary service URL.".to_owned())?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err("Boundary service must use HTTP or HTTPS.".into());
    }
    let mut url = url
        .join("lookup")
        .map_err(|_| "Invalid boundary service URL.".to_owned())?;
    url.query_pairs_mut()
        .append_pair("osm_ids", &format!("{}{}", key.osm.kind, key.osm.id))
        .append_pair("format", "geojson")
        .append_pair("polygon_geojson", "1")
        .append_pair("addressdetails", "0")
        .append_pair("polygon_threshold", "0.00001");
    Ok(url.into())
}

pub fn poll(ctx: &Context) {
    let now = Instant::now();
    let mut guard = service().lock().unwrap();
    let Some(job) = guard.next(now) else {
        if guard.active.is_some() || !guard.queue.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        return;
    };
    drop(guard);
    let url = match lookup_url(&job.key) {
        Ok(url) => url,
        Err(error) => {
            finish(&mut service().lock().unwrap(), &job.key, Err(error));
            ctx.request_repaint();
            return;
        }
    };
    let mut request = ehttp::Request::get(url);
    request.timeout = Some(Duration::from_secs(20));
    #[cfg(not(target_arch = "wasm32"))]
    request.headers.insert(
        "User-Agent",
        "LEO-Viz/0.1 (https://github.com/LeoDOS-Project/leo-viz)",
    );
    let ctx = ctx.clone();
    ehttp::fetch(request, move |response| {
        let result = response
            .map_err(|_| "Could not reach the boundary service.".to_owned())
            .and_then(|response| {
                if response.status == 429 {
                    Err("Boundary service is busy. Please try again later.".into())
                } else if !response.ok {
                    Err(format!(
                        "Boundary lookup failed (HTTP {}).",
                        response.status
                    ))
                } else {
                    parse(&response.bytes, job.key.osm, job.anchor)
                }
            });
        let mut guard = service().lock().unwrap();
        if guard
            .active
            .as_ref()
            .is_some_and(|(key, start)| key == &job.key && *start == now)
        {
            finish(&mut guard, &job.key, result);
        }
        drop(guard);
        ctx.request_repaint();
    });
}

fn finish(service: &mut Service, key: &Key, result: Result) {
    if let Some(ticket) = service.entries.get(key) {
        *ticket.lock().unwrap() = Status::Ready(result);
    }
    service.active = None;
    service.completed.push_back(key.clone());
    while service.completed.len() > CACHE_LIMIT {
        if let Some(old) = service.completed.pop_front() {
            service.entries.remove(&old);
        }
    }
}

#[derive(Deserialize)]
struct Collection {
    features: Vec<Feature>,
}

#[derive(Deserialize)]
struct Feature {
    properties: Properties,
    geometry: serde_json::Value,
}

#[derive(Deserialize)]
struct Properties {
    osm_type: String,
    osm_id: u64,
}

fn parse(bytes: &[u8], osm: OsmId, anchor: [f64; 2]) -> Result {
    if bytes.len() > 8 * 1024 * 1024 {
        return Err("Mapped boundary is too large.".into());
    }
    let collection: Collection = serde_json::from_slice(bytes)
        .map_err(|_| "Boundary service returned invalid GeoJSON.".to_owned())?;
    if collection.features.is_empty() {
        return Ok(None);
    }
    let feature = collection
        .features
        .into_iter()
        .find(|f| OsmId::new(&f.properties.osm_type, f.properties.osm_id) == Some(osm))
        .ok_or_else(|| "Boundary service returned a different place.".to_owned())?;
    match feature
        .geometry
        .get("type")
        .and_then(serde_json::Value::as_str)
    {
        Some("Polygon" | "MultiPolygon") => {
            let preview = Preview::from_geojson(feature.geometry, anchor, [255, 210, 64])?;
            let extent = preview
                .geographic_bounds()
                .ok_or_else(|| "Empty mapped boundary.".to_owned())?;
            Ok(Some(Arc::new(Area {
                preview: Arc::new(preview),
                extent,
            })))
        }
        Some("Point" | "MultiPoint" | "LineString" | "MultiLineString") | None => Ok(None),
        _ => Err("Unsupported mapped boundary geometry.".into()),
    }
}

#[cfg(test)]
pub fn test_geometry() -> serde_json::Value {
    serde_json::json!({"type":"MultiPolygon","coordinates":[
        [[[18.069,59.329],[18.071,59.329],[18.0705,59.331],[18.0695,59.3305],[18.069,59.329]],
         [[18.0698,59.3296],[18.0702,59.3296],[18.0702,59.3300],[18.0698,59.3300],[18.0698,59.3296]]],
        [[[18.072,59.329],[18.0723,59.329],[18.072,59.3293],[18.072,59.329]]]
    ]})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(id: u64) -> Key {
        Key {
            endpoint: DEFAULT_ENDPOINT.into(),
            osm: OsmId::new("way", id).unwrap(),
        }
    }
    fn response(geometry: serde_json::Value) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({"features":[{
            "properties":{"osm_type":"way","osm_id":123},"geometry":geometry,
        }]}))
        .unwrap()
    }
    #[test]
    fn preserves_irregular_polygons_holes_and_separate_islands() {
        let result = parse(&response(test_geometry()), key(123).osm, [18.07, 59.33])
            .unwrap()
            .unwrap();
        assert_eq!(result.preview.count, 1);
        assert_eq!(result.extent, [18.069, 59.331, 18.0723, 59.329]);
        assert!(result.preview.aois[0].area_km2 > 0.0);
        let original = result.preview.aois[0].geometry_hash;
        let original_area = result.preview.mean_area;
        let frame = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let transform = egui_plot::PlotTransform::new(
            frame,
            egui_plot::PlotBounds::from_min_max([18.068, 59.328], [18.073, 59.332]),
            [false, false],
        );
        let overlay = crate::aoi::preview::Overlay::projected(
            Arc::clone(&result.preview),
            crate::projection::ProjectionKind::Equirectangular,
            1.0,
        );
        for (lon, lat, inside) in [
            (18.0695, 59.3293, true),
            (18.07, 59.3298, false),
            (18.07205, 59.32905, true),
            (18.0715, 59.3295, false),
        ] {
            let screen = transform.position_from_point(&egui_plot::PlotPoint::new(lon, lat));
            assert_eq!(
                !overlay.pick(screen, &transform).is_empty(),
                inside,
                "{lon},{lat}"
            );
        }
        let mut no_hole = test_geometry();
        no_hole["coordinates"][0].as_array_mut().unwrap().pop();
        let result = parse(&response(no_hole), key(123).osm, [18.07, 59.33])
            .unwrap()
            .unwrap();
        assert_ne!(original, result.preview.aois[0].geometry_hash);
        assert!(result.preview.mean_area > original_area);
    }
    #[test]
    fn point_and_line_results_never_become_boxes() {
        for geometry in [
            serde_json::json!({"type":"Point","coordinates":[18.07,59.33]}),
            serde_json::json!({"type":"LineString","coordinates":[[18.0,59.0],[18.1,59.1]]}),
        ] {
            assert!(parse(&response(geometry), key(123).osm, [18.07, 59.33])
                .unwrap()
                .is_none());
        }
        assert!(parse(br#"{"features":[]}"#, key(123).osm, [18.07, 59.33])
            .unwrap()
            .is_none());
    }
    #[test]
    fn malformed_or_wrong_place_geometry_is_rejected() {
        assert!(parse(b"not json", key(123).osm, [18.07, 59.33]).is_err());
        assert!(parse(&response(test_geometry()), key(124).osm, [18.07, 59.33]).is_err());
        for geometry in [
            serde_json::json!({"type":"Polygon","coordinates":[]}),
            serde_json::json!({"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[2,2]]]}),
            serde_json::json!({"type":"Polygon","coordinates":[[[0,0],[181,0],[1,1],[0,0]]]}),
        ] {
            assert!(parse(&response(geometry), key(123).osm, [18.07, 59.33]).is_err());
        }
    }

    #[test]
    fn boundaries_crossing_the_dateline_keep_local_geometry_and_area() {
        let geometry = serde_json::json!({"type":"Polygon","coordinates":[[
            [179.8,1.0],[-179.8,1.0],[-179.8,2.0],[179.8,2.0],[179.8,1.0]
        ]]});
        let area = parse(&response(geometry), key(123).osm, [179.9, 1.5])
            .unwrap()
            .unwrap();
        assert!((4000.0..6000.0).contains(&area.preview.mean_area));
        let overlay = crate::aoi::preview::Overlay::projected(
            Arc::clone(&area.preview),
            crate::projection::ProjectionKind::Equirectangular,
            1.0,
        );
        let transform = egui_plot::PlotTransform::new(
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 400.0)),
            egui_plot::PlotBounds::from_min_max([-180.0, -90.0], [180.0, 90.0]),
            [false, false],
        );
        for (lon, inside) in [(179.9, true), (-179.9, true), (0.0, false)] {
            let screen = transform.position_from_point(&egui_plot::PlotPoint::new(lon, 1.5));
            assert_eq!(
                !overlay.pick(screen, &transform).is_empty(),
                inside,
                "longitude {lon}"
            );
        }
    }
    #[test]
    fn url_uses_the_selected_osm_reference_and_supports_a_proxy() {
        let mut k = key(123);
        k.endpoint = "https://example.org/geocoding/".into();
        let url = url::Url::parse(&lookup_url(&k).unwrap()).unwrap();
        assert_eq!(url.path(), "/geocoding/lookup");
        assert_eq!(
            url.query_pairs().find(|(k, _)| k == "osm_ids").unwrap().1,
            "W123"
        );
        assert_eq!(
            url.query_pairs()
                .find(|(k, _)| k == "polygon_geojson")
                .unwrap()
                .1,
            "1"
        );
        k.endpoint = "file:///etc/".into();
        assert!(lookup_url(&k).is_err());
    }
    #[test]
    fn requests_deduplicate_serialize_and_respect_the_interval() {
        let mut service = Service::default();
        let now = Instant::now();
        let first = service.request(key(1), [0.0, 0.0]);
        let duplicate = service.request(key(1), [0.0, 0.0]);
        let second = service.request(key(2), [0.0, 0.0]);
        assert!(Arc::ptr_eq(&first, &duplicate));
        assert_eq!(service.queue.len(), 2);
        assert_eq!(service.next(now).unwrap().key.osm.id, 1);
        assert!(service.next(now + Duration::from_secs(2)).is_none());
        finish(&mut service, &key(1), Ok(None));
        assert!(service.next(now + Duration::from_millis(500)).is_none());
        assert_eq!(service.next(now + INTERVAL).unwrap().key.osm.id, 2);
        assert!(matches!(*first.lock().unwrap(), Status::Ready(Ok(None))));
        assert!(matches!(*second.lock().unwrap(), Status::Pending));
    }
    #[test]
    fn cancelled_work_is_skipped_timeouts_unblock_and_errors_can_retry() {
        let mut service = Service::default();
        let now = Instant::now();
        drop(service.request(key(1), [0.0, 0.0]));
        let first = service.request(key(2), [0.0, 0.0]);
        let second = service.request(key(3), [0.0, 0.0]);
        assert_eq!(service.next(now).unwrap().key.osm.id, 2);
        assert_eq!(service.next(now + TIMEOUT).unwrap().key.osm.id, 3);
        assert!(matches!(*first.lock().unwrap(), Status::Ready(Err(_))));
        let retried = service.request(key(2), [0.0, 0.0]);
        assert!(!Arc::ptr_eq(&first, &retried));
        finish(&mut service, &key(3), Ok(None));
        assert_eq!(
            service.next(now + TIMEOUT + INTERVAL).unwrap().key.osm.id,
            2
        );
        assert!(matches!(*second.lock().unwrap(), Status::Ready(Ok(None))));
    }
    #[test]
    fn cached_results_are_bounded_and_reused() {
        let mut service = Service::default();
        for id in 1..=40 {
            let ticket = service.request(key(id), [0.0, 0.0]);
            finish(&mut service, &key(id), Ok(None));
            assert!(Arc::ptr_eq(&ticket, &service.request(key(id), [0.0, 0.0])));
        }
        assert_eq!(service.completed.len(), CACHE_LIMIT);
        assert_eq!(service.entries.len(), CACHE_LIMIT);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "calls the public Nominatim API once"]
    fn public_service_returns_a_real_stadium_outline_and_allows_browser_requests() {
        let key = key(38862723);
        let mut request = ehttp::Request::get(lookup_url(&key).unwrap());
        request.headers.insert(
            "User-Agent",
            "LEO-Viz/0.1 (https://github.com/LeoDOS-Project/leo-viz)",
        );
        request.headers.insert("Origin", "http://localhost:8080");
        let response = ehttp::fetch_blocking(&request).unwrap();
        assert!(response.ok, "HTTP {}", response.status);
        assert!(matches!(
            response.headers.get("access-control-allow-origin"),
            Some("*" | "http://localhost:8080")
        ));
        let area = parse(&response.bytes, key.osm, [13.2398144, 52.5145846])
            .unwrap()
            .unwrap();
        let [west, north, east, south] = area.extent;
        let rectangle = Preview::from_geojson(
            serde_json::json!({"type":"Polygon","coordinates":[[
                [west,south],[east,south],[east,north],[west,north],[west,south]
            ]]}),
            [13.2398144, 52.5145846],
            [255, 210, 64],
        )
        .unwrap();
        assert!(area.preview.mean_area < rectangle.mean_area * 0.98);
        assert_ne!(
            area.preview.aois[0].geometry_hash,
            rectangle.aois[0].geometry_hash
        );
    }
}
