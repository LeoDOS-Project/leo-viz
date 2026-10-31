//! Per-view Earth search using Photon's public OpenStreetMap geocoder.

use crate::{celestial::CelestialBody, config::TabConfig, projection::ProjectionKind};
use egui::{Context, Id, Rect};
use nalgebra::{Matrix3, Vector3};
use serde::Deserialize;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use web_time::Instant;

const ENDPOINT: &str = "https://photon.komoot.io/api/";
const RESULT_LIMIT: usize = 6;
const CACHE_LIMIT: usize = 32;
type SearchResult = Result<Vec<Place>, String>;
type Mailbox = Arc<Mutex<Option<SearchResult>>>;

#[derive(Clone)]
pub struct Place {
    pub label: String,
    pub lat: f64,
    pub lon: f64,
    extent: Option<[f64; 4]>,
    radius_km: f64,
    title: String,
    osm: Option<crate::location_boundary::OsmId>,
    pub area: Option<Arc<crate::location_boundary::Area>>,
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub rotation: Matrix3<f64>,
    pub zoom: f64,
}

impl Place {
    pub fn camera(&self, aspect: f64) -> Camera {
        let mut radius = self.radius_km / CelestialBody::Earth.radius_km();
        if let Some([west, north, east, south]) = self.extent {
            radius = [north, south]
                .into_iter()
                .flat_map(|lat| [west, east].map(|lon| (lat, lon)))
                .map(|(lat, lon)| {
                    let a = self.lat.to_radians();
                    let b = lat.to_radians();
                    (a.sin() * b.sin() + a.cos() * b.cos() * (lon - self.lon).to_radians().cos())
                        .clamp(-1.0, 1.0)
                        .acos()
                })
                .fold(radius, f64::max);
        }
        let zoom = 1.15 * aspect.clamp(0.1, 1.0)
            / (radius * 1.3)
                .clamp(0.00001, std::f64::consts::FRAC_PI_2)
                .sin();
        // The map texture uses latitude on an oblate surface, not a sphere.
        // Aim at that surface point so addresses remain centered at street zoom.
        let lat = self.lat.to_radians();
        let camera_lat = ((1.0 - CelestialBody::Earth.flattening()) * lat.sin()).atan2(lat.cos());
        Camera {
            rotation: crate::math::lat_lon_to_matrix(camera_lat, self.lon.to_radians()),
            zoom: zoom.clamp(1.0, crate::math::MAX_CAMERA_ZOOM),
        }
    }

    pub fn map_bounds(&self, kind: ProjectionKind, aspect: f64) -> [f64; 4] {
        let proj = kind.instance();
        let (xmin, xmax) = proj.x_range();
        let (ymin, ymax) = proj.y_range();
        let full = [xmin, xmax, ymin, ymax];
        let Some((cx, cy)) = proj.project(self.lat, self.lon) else {
            return full;
        };
        let degrees = (self.radius_km / CelestialBody::Earth.radius_km()).to_degrees();
        let [west, north, east, south] = self.extent.unwrap_or([
            self.lon - degrees / self.lat.to_radians().cos().max(0.05),
            (self.lat + degrees).min(89.9),
            self.lon + degrees / self.lat.to_radians().cos().max(0.05),
            (self.lat - degrees).max(-89.9),
        ]);
        if west < -180.0 || east > 180.0 || east < west || east - west > 180.0 {
            return full;
        }
        let mut half_x = (xmax - xmin) * 0.00001;
        let mut half_y = (ymax - ymin) * 0.00001;
        for lat in [south, self.lat, north] {
            for lon in [west, self.lon, east] {
                if let Some((x, y)) = proj.project(lat, lon) {
                    half_x = half_x.max((x - cx).abs());
                    half_y = half_y.max((y - cy).abs());
                }
            }
        }
        half_y = (half_y * 1.3).max(half_x * 1.3 / aspect.max(0.1));
        half_x = half_y * aspect.max(0.1);
        [cx - half_x, cx + half_x, cy - half_y, cy + half_y]
    }

    pub fn paint_globe(
        &self,
        painter: &egui::Painter,
        transform: &egui_plot::PlotTransform,
        rotation: &Matrix3<f64>,
        radius: f64,
        flattening: f64,
    ) {
        let project = |[lon, lat]: [f64; 2]| {
            let (lat, lon) = (lat.to_radians(), lon.to_radians());
            crate::globe_overlay::project_surface(
                Vector3::new(lat.cos() * lon.cos(), lat.sin(), -lat.cos() * lon.sin()),
                rotation,
                radius,
                1.0 - flattening,
            )
        };
        let screen = |point: Vector3<f64>| {
            transform.position_from_point(&egui_plot::PlotPoint::new(point.x, point.y))
        };
        let painter = painter.with_clip_rect(*transform.frame());
        let (point, front) = project([self.lon, self.lat]);
        if front >= 0.0 {
            self.paint_marker(&painter, screen(point), *transform.frame());
        }
    }

    pub fn paint_map(
        &self,
        painter: &egui::Painter,
        transform: &egui_plot::PlotTransform,
        projection: &dyn crate::projection::Projection,
    ) {
        let painter = painter.with_clip_rect(*transform.frame());
        if let Some((x, y)) = projection.project(self.lat, self.lon) {
            let point = transform.position_from_point(&egui_plot::PlotPoint::new(x, y));
            self.paint_marker(&painter, point, *transform.frame());
        }
    }

    fn paint_marker(&self, painter: &egui::Painter, point: egui::Pos2, frame: Rect) {
        if !frame.contains(point) || frame.width() < 30.0 || frame.height() < 30.0 {
            return;
        }
        painter.circle_filled(point, 4.0, HIGHLIGHT_COLOR);
        painter.circle_stroke(point, 8.0, egui::Stroke::new(2.0_f32, HIGHLIGHT_COLOR));
        let galley = painter.layout(
            self.title.clone(),
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
            (frame.width() - 20.0).min(300.0),
        );
        let text_pos = egui::pos2(
            (point.x - galley.size().x / 2.0).clamp(
                frame.left() + 5.0,
                (frame.right() - galley.size().x - 5.0).max(frame.left() + 5.0),
            ),
            (point.y - galley.size().y - 14.0).clamp(
                frame.top() + 5.0,
                (frame.bottom() - galley.size().y - 5.0).max(frame.top() + 5.0),
            ),
        );
        painter.rect_filled(
            Rect::from_min_size(text_pos, galley.size()).expand(3.0),
            3.0,
            egui::Color32::from_black_alpha(220),
        );
        painter.galley(text_pos, galley, egui::Color32::WHITE);
    }
}

const HIGHLIGHT_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 210, 64);

#[derive(Deserialize)]
struct FeatureCollection {
    features: Vec<Feature>,
}

#[derive(Deserialize)]
struct Feature {
    geometry: Geometry,
    properties: Properties,
}

#[derive(Deserialize)]
struct Geometry {
    coordinates: Vec<f64>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Properties {
    osm_type: String,
    osm_id: Option<u64>,
    name: String,
    street: String,
    housenumber: String,
    city: String,
    district: String,
    state: String,
    country: String,
    postcode: String,
    #[serde(rename = "type")]
    kind: String,
    extent: Option<[f64; 4]>,
}

fn parse(bytes: &[u8]) -> SearchResult {
    let data: FeatureCollection = serde_json::from_slice(bytes)
        .map_err(|_| "The search service returned an invalid response.".to_owned())?;
    Ok(data
        .features
        .into_iter()
        .filter_map(|feature| {
            let [lon, lat] = feature.geometry.coordinates.as_slice() else {
                return None;
            };
            if !lat.is_finite() || !lon.is_finite() || lat.abs() > 90.0 || lon.abs() > 180.0 {
                return None;
            }
            let p = feature.properties;
            let mut parts = Vec::new();
            let street = format!("{} {}", p.street, p.housenumber).trim().to_owned();
            for part in [
                p.name, street, p.city, p.district, p.state, p.postcode, p.country,
            ] {
                if !part.is_empty() && !parts.contains(&part) {
                    parts.push(part);
                }
            }
            if parts.is_empty() {
                return None;
            }
            let extent = p.extent.filter(|e| {
                e.iter().all(|v| v.is_finite())
                    && e[0].abs() <= 180.0
                    && e[2].abs() <= 180.0
                    && e[1].abs() <= 90.0
                    && e[3].abs() <= 90.0
                    && e[1] > e[3]
                    && e[0] != e[2]
            });
            let radius_km = match p.kind.as_str() {
                "country" => 1000.0,
                "state" => 200.0,
                "county" => 50.0,
                "city" => 15.0,
                "district" | "locality" => 5.0,
                _ => 1.0,
            };
            Some(Place {
                title: parts[0].clone(),
                osm: p
                    .osm_id
                    .and_then(|id| crate::location_boundary::OsmId::new(&p.osm_type, id)),
                area: None,
                label: parts.join(", "),
                lat: *lat,
                lon: *lon,
                extent,
                radius_km,
            })
        })
        .take(RESULT_LIMIT)
        .collect())
}

fn search_url(query: &str) -> String {
    let mut url = url::Url::parse(ENDPOINT).expect("valid Photon endpoint");
    url.query_pairs_mut()
        .append_pair("q", query)
        .append_pair("limit", &RESULT_LIMIT.to_string())
        .append_pair("lang", "en");
    url.into()
}

#[derive(Default)]
struct Service {
    last_request: Option<Instant>,
    cache: VecDeque<(String, Vec<Place>)>,
}

fn service() -> &'static Mutex<Service> {
    static SERVICE: OnceLock<Mutex<Service>> = OnceLock::new();
    SERVICE.get_or_init(Mutex::default)
}

fn request(query: String, ctx: Context) -> Mailbox {
    let mailbox = Mailbox::default();
    let mut guard = service().lock().unwrap();
    if let Some((_, results)) = guard.cache.iter().find(|(q, _)| q == &query) {
        *mailbox.lock().unwrap() = Some(Ok(results.clone()));
        return mailbox;
    }
    if guard
        .last_request
        .is_some_and(|last| last.elapsed() < Duration::from_secs(1))
    {
        *mailbox.lock().unwrap() = Some(Err("Please wait a second before searching again.".into()));
        return mailbox;
    }
    guard.last_request = Some(Instant::now());
    drop(guard);
    let output = Arc::clone(&mailbox);
    let mut request = ehttp::Request::get(search_url(&query));
    request.timeout = Some(Duration::from_secs(20));
    ehttp::fetch(request, move |response| {
        let result = response
            .map_err(|_| "Could not reach the search service. Please try again.".to_owned())
            .and_then(|response| {
                if response.status == 429 {
                    Err("The search service is busy. Please try again later.".into())
                } else if !response.ok {
                    Err(format!(
                        "Search failed (HTTP {}). Please try again.",
                        response.status
                    ))
                } else {
                    parse(&response.bytes)
                }
            });
        if let Ok(places) = &result {
            let mut service = service().lock().unwrap();
            service.cache.push_back((query, places.clone()));
            while service.cache.len() > CACHE_LIMIT {
                service.cache.pop_front();
            }
        }
        *output.lock().unwrap() = Some(result);
        ctx.request_repaint();
    });
    mailbox
}

#[derive(Default)]
pub struct Search {
    pub open: bool,
    pub camera: Option<Camera>,
    viewport: Option<Rect>,
    last_frame: u64,
    focus: bool,
    query: String,
    submitted: Option<String>,
    results: Vec<Place>,
    active_result: Option<usize>,
    scroll_to_active: bool,
    last_pointer: Option<egui::Pos2>,
    ime_composing: bool,
    selection: Option<Place>,
    hovered: Option<Place>,
    navigation: Option<Place>,
    error: Option<String>,
    pending: Option<(Instant, Mailbox)>,
    boundary: Option<crate::location_boundary::Ticket>,
    boundary_checked: bool,
    boundary_error: Option<String>,
    fit_boundary: bool,
    last_map_bounds: Option<[f64; 4]>,
}

impl Clone for Search {
    fn clone(&self) -> Self {
        Self {
            camera: self.camera,
            ..Self::default()
        }
    }
}

impl Search {
    pub fn highlight(&self) -> Option<&Place> {
        self.hovered.as_ref().or(self.selection.as_ref())
    }

    pub fn select(&mut self, place: Place) {
        self.query = place.label.clone();
        self.submitted = None;
        self.results.clear();
        self.active_result = None;
        self.scroll_to_active = false;
        self.pending = None;
        self.error = None;
        self.hovered = None;
        self.selection = Some(place.clone());
        self.navigation = Some(place);
        self.boundary = None;
        self.boundary_checked = false;
        self.boundary_error = None;
        self.fit_boundary = true;
    }

    pub fn user_navigated(&mut self) {
        self.fit_boundary = false;
    }

    pub fn observe_map(&mut self, bounds: [f64; 4], requested: bool) {
        if !requested && self.last_map_bounds.is_some_and(|old| old != bounds) {
            self.user_navigated();
        }
        self.last_map_bounds = Some(bounds);
    }

    fn poll_boundary(&mut self, ctx: &Context) {
        if !self.boundary_checked {
            if let Some(place) = &self.selection {
                self.boundary_checked = true;
                if let Some(osm) = place.osm {
                    self.boundary = Some(crate::location_boundary::request(
                        ctx,
                        osm,
                        [place.lon, place.lat],
                    ));
                }
            }
        }
        crate::location_boundary::poll(ctx);
        let status = self
            .boundary
            .as_ref()
            .map(|ticket| ticket.lock().unwrap().clone());
        if let Some(crate::location_boundary::Status::Ready(result)) = status {
            self.boundary = None;
            match result {
                Ok(area) => {
                    if let Some(place) = &mut self.selection {
                        if let Some(area) = &area {
                            place.extent = Some(area.extent);
                        }
                        place.area = area;
                        if self.fit_boundary && place.area.is_some() {
                            self.navigation = Some(place.clone());
                        }
                    }
                    self.fit_boundary = false;
                }
                Err(error) => self.boundary_error = Some(error),
            }
        }
    }

    pub fn show(&mut self, ctx: &Context, id: Id, viewport: Rect) -> Option<Place> {
        self.viewport = Some(viewport);
        self.last_frame = ctx.cumulative_frame_nr();
        self.hovered = None;
        self.poll_boundary(ctx);
        if !self.open || viewport.width() < 100.0 || viewport.height() < 60.0 {
            return self.navigation.take();
        }
        if let Some((started, pending)) = &self.pending {
            let result = pending.lock().unwrap().take();
            if let Some(result) = result {
                match result {
                    Ok(places) => {
                        self.results = places;
                        self.active_result = None;
                        self.scroll_to_active = true;
                    }
                    Err(error) => self.error = Some(error),
                }
                self.pending = None;
            } else if started.elapsed() > Duration::from_secs(25) {
                self.error = Some("Search timed out. Please try again.".into());
                self.pending = None;
            } else {
                ctx.request_repaint_after(Duration::from_millis(100));
            }
        }
        if self.results.is_empty() {
            self.active_result = None;
        } else if self
            .active_result
            .is_none_or(|index| index >= self.results.len())
        {
            self.active_result = Some(0);
        }
        let query_id = id.with("query");
        let focused = self.focus || ctx.memory(|memory| memory.has_focus(query_id));
        let menu_open = !self.results.is_empty() || self.pending.is_some();
        let mut ime_frame = self.ime_composing;
        if focused {
            ctx.input(|input| {
                for event in &input.events {
                    if let egui::Event::Ime(event) = event {
                        ime_frame = true;
                        self.ime_composing =
                            !matches!(event, egui::ImeEvent::Commit(_) | egui::ImeEvent::Disabled);
                    }
                }
            });
        } else {
            self.ime_composing = false;
        }
        let mut actions = Vec::new();
        // Enter commits composed text before it can submit or select a place.
        if focused && !ime_frame {
            ctx.input_mut(|input| {
                input.events.retain(|event| {
                    let egui::Event::Key {
                        key,
                        modifiers,
                        pressed: true,
                        repeat,
                        ..
                    } = event
                    else {
                        return true;
                    };
                    let action = match key {
                        egui::Key::Enter if modifiers.matches_exact(egui::Modifiers::NONE) => {
                            Some(SearchAction::Accept)
                        }
                        egui::Key::J
                            if menu_open && modifiers.matches_exact(egui::Modifiers::CTRL) =>
                        {
                            Some(SearchAction::Next)
                        }
                        egui::Key::K
                            if menu_open && modifiers.matches_exact(egui::Modifiers::CTRL) =>
                        {
                            Some(SearchAction::Previous)
                        }
                        egui::Key::Tab | egui::Key::ArrowDown
                            if menu_open && modifiers.matches_exact(egui::Modifiers::NONE) =>
                        {
                            Some(SearchAction::Next)
                        }
                        egui::Key::Tab
                            if menu_open && modifiers.matches_exact(egui::Modifiers::SHIFT) =>
                        {
                            Some(SearchAction::Previous)
                        }
                        egui::Key::ArrowUp
                            if menu_open && modifiers.matches_exact(egui::Modifiers::NONE) =>
                        {
                            Some(SearchAction::Previous)
                        }
                        _ => None,
                    };
                    if let Some(action) = action {
                        if !*repeat || !matches!(action, SearchAction::Accept) {
                            actions.push(action);
                        }
                        false
                    } else {
                        true
                    }
                })
            });
        }
        let mut selected = None;
        let mut want_focus = false;
        let mut keyboard_moved = false;
        let pointer = ctx.input(|input| input.pointer.hover_pos());
        let pointer_moved = pointer != self.last_pointer;
        self.last_pointer = pointer;
        let width = 330.0_f32.min(viewport.width() - 16.0);
        egui::Area::new(id)
            .order(egui::Order::Foreground)
            .fixed_pos(viewport.min + egui::vec2(8.0, 8.0))
            .default_width(width)
            .constrain_to(viewport)
            .show(ctx, |ui| {
                ui.set_width(width);
                ui.set_clip_rect(viewport.intersect(ui.clip_rect()));
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    let mut submit = false;
                    ui.horizontal(|ui| {
                        let input = ui.add_sized(
                            [
                                (ui.available_width() - 48.0 - ui.spacing().item_spacing.x * 2.0)
                                    .max(12.0),
                                24.0,
                            ],
                            egui::TextEdit::singleline(&mut self.query)
                                .id(query_id)
                                .return_key(None)
                                .lock_focus(menu_open)
                                .hint_text("Search places or addresses")
                                .char_limit(256),
                        );
                        if self.focus {
                            input.request_focus();
                            self.focus = false;
                        }
                        if input.changed() {
                            // Detach old requests so late results cannot replace the new query.
                            self.pending = None;
                            self.submitted = None;
                            self.results.clear();
                            self.active_result = None;
                            self.scroll_to_active = false;
                            self.error = None;
                            self.selection = None;
                            self.navigation = None;
                            self.boundary = None;
                            self.boundary_checked = false;
                            self.boundary_error = None;
                            self.fit_boundary = false;
                        }
                        for action in &actions {
                            match action {
                                SearchAction::Accept => {
                                    if let Some(place) =
                                        self.active_result.and_then(|index| self.results.get(index))
                                    {
                                        selected = Some(place.clone());
                                    } else {
                                        submit = true;
                                    }
                                }
                                SearchAction::Next | SearchAction::Previous
                                    if !self.results.is_empty() =>
                                {
                                    let count = self.results.len();
                                    let current = self.active_result.unwrap_or(0);
                                    self.active_result = Some(match action {
                                        SearchAction::Next => (current + 1) % count,
                                        _ => (current + count - 1) % count,
                                    });
                                    self.scroll_to_active = true;
                                    keyboard_moved = true;
                                }
                                _ => {}
                            }
                        }
                        want_focus |= !actions.is_empty();
                        if (input.has_focus() || input.lost_focus())
                            && ui.input_mut(|i| {
                                i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)
                            })
                        {
                            self.open = false;
                            input.surrender_focus();
                        }
                        submit |= ui
                            .add_enabled(
                                self.pending.is_none() && !self.query.trim().is_empty(),
                                egui::Button::image(
                                    egui::Image::new(egui::include_image!(
                                        "../assets/lucide-search.svg"
                                    ))
                                    .tint(ui.visuals().text_color())
                                    .fit_to_exact_size(egui::vec2(14.0, 14.0)),
                                )
                                .min_size(egui::vec2(24.0, 24.0)),
                            )
                            .on_hover_text("Search")
                            .clicked();
                        if ui
                            .add(
                                egui::Button::image(
                                    egui::Image::new(egui::include_image!(
                                        "../assets/lucide-x.svg"
                                    ))
                                    .tint(ui.visuals().text_color())
                                    .fit_to_exact_size(egui::vec2(14.0, 14.0)),
                                )
                                .min_size(egui::vec2(24.0, 24.0)),
                            )
                            .on_hover_text("Close search")
                            .clicked()
                        {
                            self.open = false;
                            input.surrender_focus();
                        }
                    });
                    if submit && self.pending.is_none() && !self.query.trim().is_empty() {
                        let query = self.query.trim().to_owned();
                        self.results.clear();
                        self.active_result = None;
                        self.error = None;
                        self.submitted = Some(query.clone());
                        self.pending = Some((Instant::now(), request(query, ctx.clone())));
                        want_focus = true;
                        ctx.request_repaint();
                    }
                    if self.pending.is_some() {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Searching...");
                        });
                    } else if let Some(error) = &self.error {
                        ui.colored_label(ui.visuals().error_fg_color, error);
                    } else if self.submitted.is_some() {
                        if self.results.is_empty() {
                            ui.label("No places found.");
                        } else {
                            egui::ScrollArea::vertical()
                                .max_height((viewport.height() - 100.0).max(25.0).min(260.0))
                                .show(ui, |ui| {
                                    for (index, place) in self.results.iter().enumerate() {
                                        let row =
                                            ui.add(
                                                egui::Button::selectable(
                                                    self.active_result == Some(index),
                                                    &place.label,
                                                )
                                                .wrap()
                                                .min_size(egui::vec2(
                                                    ui.available_width(),
                                                    ui.spacing().interact_size.y,
                                                )),
                                            );
                                        if row.hovered() && pointer_moved && !keyboard_moved {
                                            self.active_result = Some(index);
                                        }
                                        if self.active_result == Some(index)
                                            && self.scroll_to_active
                                        {
                                            row.scroll_to_me(Some(egui::Align::Center));
                                        }
                                        if row.clicked() {
                                            selected = Some(place.clone());
                                        }
                                    }
                                });
                            self.scroll_to_active = false;
                        }
                    }
                    if self.boundary.is_some() {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Loading boundary...");
                        });
                    } else if let Some(error) = &self.boundary_error {
                        ui.colored_label(ui.visuals().error_fg_color, error);
                        if ui.small_button("Retry").clicked() {
                            self.boundary_checked = false;
                            self.boundary_error = None;
                        }
                    } else if self.selection.as_ref().is_some_and(|p| p.area.is_none())
                        && self.boundary_checked
                    {
                        ui.small("No mapped area.");
                    }
                });
            });
        if let Some(place) = selected {
            self.select(place);
        } else if self.open {
            self.hovered = self
                .active_result
                .and_then(|index| self.results.get(index))
                .cloned();
            if let Some(place) = &mut self.hovered {
                place.area = place
                    .osm
                    .and_then(|osm| crate::location_boundary::cached(ctx, osm));
            }
        }
        if self.open && want_focus {
            ctx.memory_mut(|memory| memory.request_focus(query_id));
        }
        if self.open && ctx.memory(|memory| memory.has_focus(query_id)) {
            ctx.memory_mut(|memory| {
                memory.set_focus_lock_filter(
                    query_id,
                    egui::EventFilter {
                        tab: !self.results.is_empty() || self.pending.is_some(),
                        horizontal_arrows: true,
                        vertical_arrows: true,
                        ..Default::default()
                    },
                )
            });
            #[cfg(target_arch = "wasm32")]
            WEB_RESULTS_FOCUSED
                .with(|focused| focused.set(!self.results.is_empty() || self.pending.is_some()));
        }
        self.navigation.take()
    }
}

enum SearchAction {
    Next,
    Previous,
    Accept,
}

pub fn sources_ui(ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.strong("Geographic Data");
        ui.hyperlink_to("Photon", "https://photon.komoot.io")
            .on_hover_text("Addresses are sent to Photon's public geocoding service");
        ui.hyperlink_to("Nominatim", "https://nominatim.org");
        ui.hyperlink_to(
            "OpenStreetMap contributors",
            "https://www.openstreetmap.org/copyright",
        );
    });
    egui::CollapsingHeader::new("Boundary Service")
        .id_salt("boundary_service_settings")
        .show(ui, crate::location_boundary::settings);
}

pub fn handle_shortcut(ctx: &Context, tabs: &mut [TabConfig], active_tab: usize) {
    #[cfg(target_arch = "wasm32")]
    WEB_RESULTS_FOCUSED.with(|focused| focused.set(false));
    let eligible = |tab: &TabConfig| {
        tab.slides.is_none() && tab.settings.view_mode == crate::config::ViewMode::Planet
    };
    let has_earth = tabs.iter().any(|tab| {
        eligible(tab)
            && tab
                .planets
                .iter()
                .any(|p| p.celestial_body == CelestialBody::Earth)
    });
    #[cfg(target_arch = "wasm32")]
    WEB_SEARCH_ENABLED.with(|enabled| enabled.set(has_earth));
    if !has_earth
        || !ctx.input_mut(|i| {
            i.consume_key(egui::Modifiers::COMMAND, egui::Key::F)
                || i.consume_key(egui::Modifiers::CTRL, egui::Key::F)
        })
    {
        return;
    }
    let pointer = ctx.input(|i| i.pointer.hover_pos());
    let focus = tabs
        .iter()
        .enumerate()
        .flat_map(|(ti, tab)| {
            tab.planets.iter().enumerate().filter_map(move |(pi, p)| {
                (eligible(tab) && p.celestial_body == CelestialBody::Earth).then_some((ti, pi))
            })
        })
        .find(|&(ti, pi)| {
            pointer.is_some_and(|pos| {
                let search = &tabs[ti].planets[pi].location_search;
                search.last_frame + 1 >= ctx.cumulative_frame_nr()
                    && search.viewport.is_some_and(|r| r.contains(pos))
            })
        })
        .or_else(|| {
            tabs.get(active_tab)
                .filter(|tab| eligible(tab))
                .and_then(|tab| {
                    tab.planets
                        .iter()
                        .position(|p| p.celestial_body == CelestialBody::Earth)
                        .map(|pi| (active_tab, pi))
                })
        });
    for (ti, tab) in tabs.iter_mut().enumerate() {
        if !eligible(tab) {
            continue;
        }
        for (pi, planet) in tab.planets.iter_mut().enumerate() {
            if planet.celestial_body == CelestialBody::Earth {
                planet.location_search.open = true;
                planet.location_search.focus = focus == Some((ti, pi));
            }
        }
    }
}

pub fn navigate(tab: &mut TabConfig, planet_idx: usize, place: &Place, aspect: f64) {
    if tab.planets[planet_idx].celestial_body != CelestialBody::Earth {
        return;
    }
    let camera = place.camera(aspect);
    if tab.planets.len() == 1 {
        tab.planets[planet_idx].location_search.camera = None;
        tab.settings.rotation = camera.rotation;
        tab.settings.zoom = camera.zoom;
        tab.settings.camera_roll = 0.0;
        tab.settings.earth_fixed_camera = true;
        tab.settings.sun_fixed_camera = false;
        tab.settings.camera_mode = crate::config::CameraMode::Unlocked;
        tab.settings.auto_rotate = false;
        tab.settings.auto_zoom = false;
    } else {
        tab.planets[planet_idx].location_search.camera = Some(camera);
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn prepare_diagnostic(search: &mut Search, ctx: &Context) {
    search.open = true;
    search.query = "Olympiastadion Berlin".into();
    search.submitted = Some(search.query.clone());
    search.pending = Some((Instant::now(), request(search.query.clone(), ctx.clone())));
}

#[cfg(not(target_arch = "wasm32"))]
impl Search {
    pub fn diagnostic_boundary_ready(&self, projection: ProjectionKind) -> bool {
        self.selection
            .as_ref()
            .and_then(|place| place.area.as_ref())
            .is_some_and(|area| {
                projection == ProjectionKind::Orthographic
                    || area.preview.projected_geometry_ready(projection)
            })
    }

    pub fn diagnostic_area(&self) -> Option<Place> {
        self.results
            .iter()
            .find(|place| place.extent.is_some())
            .cloned()
    }

    pub fn diagnostic_state(&self) -> serde_json::Value {
        serde_json::json!({
            "query": self.query, "results": self.results.len(), "pending": self.pending.is_some(),
            "active_result": self.active_result,
            "error": self.error,
            "selection": self.selection.as_ref().map(|place| serde_json::json!({
                "label": place.label, "lat": place.lat, "lon": place.lon, "extent": place.extent,
                "mapped_area": place.area.is_some(),
                "area_km2": place.area.as_ref().map(|area| area.preview.mean_area),
            })),
            "boundary_pending": self.boundary.is_some(), "boundary_error": self.boundary_error,
        })
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_SEARCH_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static WEB_RESULTS_FOCUSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(target_arch = "wasm32")]
pub fn install_web_shortcut(document: &web_sys::Document) {
    use wasm_bindgen::{closure::Closure, JsCast};
    let handler =
        Closure::<dyn Fn(web_sys::KeyboardEvent)>::new(|event: web_sys::KeyboardEvent| {
            let search = event.key().eq_ignore_ascii_case("f")
                && (event.meta_key() || event.ctrl_key())
                && !event.alt_key()
                && !event.shift_key()
                && WEB_SEARCH_ENABLED.with(|enabled| enabled.get());
            let results = matches!(event.key().to_ascii_lowercase().as_str(), "j" | "k")
                && event.ctrl_key()
                && !event.meta_key()
                && !event.alt_key()
                && !event.shift_key()
                && WEB_RESULTS_FOCUSED.with(|focused| focused.get());
            if search || results {
                event.prevent_default();
            }
        });
    document
        .add_event_listener_with_callback_and_bool(
            "keydown",
            handler.as_ref().unchecked_ref(),
            true,
        )
        .expect("install Earth search shortcut");
    handler.forget();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PlanetConfig;
    use egui_plot::PlotItem;

    const FIXTURE: &[u8] = br#"{"features":[
        {"geometry":{"coordinates":[18.0710935,59.3251172]},
         "properties":{"name":"Stockholm","city":"Stockholm","country":"Sweden","type":"city"}},
        {"geometry":{"coordinates":[18.07,59.33]},
         "properties":{"street":"Sveavagen","housenumber":"17","city":"Stockholm","country":"Sweden",
                       "extent":[18.069,59.331,18.071,59.329]}}
    ]}"#;

    fn tab() -> TabConfig {
        let mut tab = TabConfig::new("Search test".into());
        tab.planets = vec![
            PlanetConfig::new("Earth 1".into()),
            PlanetConfig::new("Earth 2".into()),
        ];
        tab
    }

    fn mapped_place() -> Place {
        let mut place = parse(FIXTURE).unwrap().remove(1);
        let preview = crate::aoi::preview::Preview::from_geojson(
            crate::location_boundary::test_geometry(),
            [place.lon, place.lat],
            [255, 210, 64],
        )
        .unwrap();
        let extent = preview.geographic_bounds().unwrap();
        place.area = Some(Arc::new(crate::location_boundary::Area {
            preview: Arc::new(preview),
            extent,
        }));
        place
    }

    fn key(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }
    }

    fn find_text(shape: &egui::Shape, label: &str) -> Option<Rect> {
        match shape {
            egui::Shape::Text(text) if text.galley.job.text == label => {
                Some(Rect::from_min_size(text.pos, text.galley.size()))
            }
            egui::Shape::Vec(shapes) => shapes.iter().find_map(|shape| find_text(shape, label)),
            _ => None,
        }
    }

    fn frame(ctx: &Context, tabs: &mut [TabConfig], events: Vec<egui::Event>) -> egui::FullOutput {
        egui_extras::install_image_loaders(ctx);
        ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 400.0),
                )),
                events,
                ..Default::default()
            },
            |ui| {
                let ctx = ui.ctx();
                handle_shortcut(ctx, tabs, 0);
                let mut selections = Vec::new();
                for (index, planet) in tabs[0].planets.iter_mut().enumerate() {
                    if planet.celestial_body == CelestialBody::Earth {
                        if let Some(place) = planet.location_search.show(
                            ctx,
                            Id::new(index),
                            Rect::from_min_size(
                                egui::pos2(index as f32 * 400.0, 0.0),
                                egui::vec2(400.0, 400.0),
                            ),
                        ) {
                            selections.push((index, place));
                        }
                    }
                }
                for (index, place) in selections {
                    navigate(&mut tabs[0], index, &place, 1.0);
                }
            },
        )
    }

    #[test]
    fn parses_addresses_and_extent_without_duplicate_city() {
        let places = parse(FIXTURE).unwrap();
        assert_eq!(places.len(), 2);
        assert_eq!(places[0].label, "Stockholm, Sweden");
        assert_eq!(places[0].lat, 59.3251172);
        assert_eq!(places[0].lon, 18.0710935);
        assert_eq!(places[1].label, "Sveavagen 17, Stockholm, Sweden");
        assert_eq!(places[1].extent, Some([18.069, 59.331, 18.071, 59.329]));
    }

    #[test]
    fn rejects_invalid_coordinates_and_malformed_responses() {
        assert!(parse(b"not json").is_err());
        assert!(parse(br#"{"features":[]}"#).unwrap().is_empty());
        assert!(parse(
            br#"{"features":[
            {"geometry":{"coordinates":[181,0]},"properties":{"name":"Invalid"}},
            {"geometry":{"coordinates":[0,-91]},"properties":{"name":"Invalid"}},
            {"geometry":{"coordinates":[]},"properties":{"name":"Invalid"}},
            {"geometry":{"coordinates":[0,0]},"properties":{}}
        ]}"#
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn query_encoding_preserves_unicode_and_reserved_characters() {
        let query = "Björkgatan 3 & café? #Stockholm";
        let url = url::Url::parse(&search_url(query)).unwrap();
        assert_eq!(url.host_str(), Some("photon.komoot.io"));
        assert_eq!(
            url.query_pairs().find(|(key, _)| key == "q").unwrap().1,
            query
        );
        assert_eq!(url.query_pairs().count(), 3);
    }

    #[test]
    fn globe_navigation_centers_the_location_and_fits_the_area() {
        let places = parse(FIXTURE).unwrap();
        let camera = places[0].camera(1.5);
        let (lat, lon) = (places[0].lat.to_radians(), places[0].lon.to_radians());
        let target =
            nalgebra::Vector3::new(lat.cos() * lon.cos(), lat.sin(), -lat.cos() * lon.sin());
        let (centered, front) = crate::globe_overlay::project_surface(
            target,
            &camera.rotation,
            CelestialBody::Earth.radius_km(),
            1.0 - CelestialBody::Earth.flattening(),
        );
        assert!(front > 0.0);
        assert!(centered.x.abs() < 1e-8 && centered.y.abs() < 1e-8);
        assert!(places[1].camera(1.5).zoom > camera.zoom);
        assert!(places[0].camera(0.5).zoom < camera.zoom);
    }

    #[test]
    fn globe_search_center_matches_imagery_at_street_zoom_and_both_hemispheres() {
        let mut place = parse(FIXTURE).unwrap().remove(1);
        place.extent = None;
        for lat in [-89.99, -59.33, -30.0, 0.0, 30.0, 59.33, 89.99] {
            for lon in [-179.99, 18.07, 179.99] {
                place.lat = lat;
                place.lon = lon;
                let camera = place.camera(1.0);
                assert!(camera.zoom > 1000.0);
                for flattening in [
                    CelestialBody::Earth.flattening(),
                    f64::from(CelestialBody::Earth.flattening() as f32),
                ] {
                    let (picked_lat, picked_lon) = crate::globe_overlay::pick_surface(
                        0.0,
                        0.0,
                        &camera.rotation,
                        CelestialBody::Earth.radius_km(),
                        1.0 - flattening,
                    )
                    .unwrap();
                    assert!(
                        (picked_lat - lat).abs() < 1e-8,
                        "latitude {lat}: {picked_lat}"
                    );
                    assert!(((picked_lon - lon + 180.0).rem_euclid(360.0) - 180.0).abs() < 1e-8);
                }
            }
        }
    }

    #[test]
    fn returned_extent_is_not_treated_as_a_mapped_area() {
        let places = parse(FIXTURE).unwrap();
        assert!(places[1].extent.is_some());
        assert!(places.iter().all(|place| place.area.is_none()));
    }

    #[test]
    fn highlights_render_at_the_real_globe_center_and_are_clipped_on_the_far_side() {
        let place = mapped_place();
        let camera = place.camera(1.0);
        let ctx = Context::default();
        let frame = Rect::from_min_size(egui::pos2(30.0, 40.0), egui::vec2(500.0, 500.0));
        let margin = CelestialBody::Earth.radius_km() * 1.15 / camera.zoom;
        let transform = egui_plot::PlotTransform::new(
            frame,
            egui_plot::PlotBounds::from_min_max([-margin, -margin], [margin, margin]),
            [false, false],
        );
        let render = |rotation: Matrix3<f64>| {
            ctx.run_ui(egui::RawInput::default(), |ui| {
                let mut shapes = Vec::new();
                crate::aoi::preview::Overlay::new(
                    Arc::clone(&place.area.as_ref().unwrap().preview),
                    rotation,
                    CelestialBody::Earth.radius_km(),
                    CelestialBody::Earth.flattening(),
                    1.0,
                )
                .shapes(ui, &transform, &mut shapes);
                for shape in shapes {
                    ui.painter().add(shape);
                }
                place.paint_globe(
                    ui.painter(),
                    &transform,
                    &rotation,
                    CelestialBody::Earth.radius_km(),
                    CelestialBody::Earth.flattening(),
                );
            })
        };
        let output = render(camera.rotation);
        let marker = output
            .shapes
            .iter()
            .find_map(|shape| match &shape.shape {
                egui::Shape::Circle(circle) if circle.fill == HIGHLIGHT_COLOR => {
                    Some(circle.center)
                }
                _ => None,
            })
            .expect("location marker");
        assert!(marker.distance(frame.center()) < 0.01);
        assert!(output.shapes.iter().any(
            |shape| matches!(&shape.shape, egui::Shape::Mesh(mesh) if !mesh.vertices.is_empty())
        ));
        let reverse = Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        let output = render(reverse * camera.rotation);
        assert!(!output
            .shapes
            .iter()
            .any(|shape| matches!(&shape.shape, egui::Shape::Circle(_))));
        assert!(!output.shapes.iter().any(
            |shape| matches!(&shape.shape, egui::Shape::Mesh(mesh) if !mesh.vertices.is_empty())
        ));
    }

    #[test]
    fn navigation_changes_only_the_selected_earth_view() {
        let mut tab = tab();
        let old_rotation = tab.settings.rotation;
        let old_zoom = tab.settings.zoom;
        navigate(&mut tab, 1, &parse(FIXTURE).unwrap()[0], 1.0);
        assert!(tab.planets[0].location_search.camera.is_none());
        assert!(tab.planets[1].location_search.camera.is_some());
        assert_eq!(tab.settings.rotation, old_rotation);
        assert_eq!(tab.settings.zoom, old_zoom);
        tab.planets[0].celestial_body = CelestialBody::Mars;
        navigate(&mut tab, 0, &parse(FIXTURE).unwrap()[0], 1.0);
        assert!(tab.planets[0].location_search.camera.is_none());
    }

    #[test]
    fn single_view_navigation_stops_tracking_and_keeps_camera_controls_working() {
        let mut tab = tab();
        tab.planets.truncate(1);
        tab.settings.auto_rotate = true;
        tab.settings.auto_zoom = true;
        tab.settings.sun_fixed_camera = true;
        navigate(&mut tab, 0, &parse(FIXTURE).unwrap()[0], 1.0);
        assert!(tab.settings.zoom > 100.0);
        assert!(tab.settings.earth_fixed_camera);
        assert!(
            !tab.settings.sun_fixed_camera && !tab.settings.auto_rotate && !tab.settings.auto_zoom
        );
        assert!(tab.planets[0].location_search.camera.is_none());
    }

    #[test]
    fn map_navigation_centers_results_in_all_projections() {
        use ProjectionKind::*;
        let place = mapped_place();
        for projection in [
            Equirectangular,
            Mercator,
            Mollweide,
            Sinusoidal,
            AzimuthalEquidistant,
            Hammer,
            HEALPix,
            Cassini,
            TransverseMercator,
            LambertAzimuthalEqualArea,
            GallPeters,
            PeirceQuincuncial,
        ] {
            for aspect in [0.5, 1.5] {
                let bounds = place.map_bounds(projection, aspect);
                assert!(
                    bounds.iter().all(|v| v.is_finite()),
                    "{}",
                    projection.label()
                );
                assert!(bounds[0] < bounds[1] && bounds[2] < bounds[3]);
                let (x, y) = projection.instance().project(place.lat, place.lon).unwrap();
                assert!((x - (bounds[0] + bounds[1]) / 2.0).abs() < 1e-10);
                assert!((y - (bounds[2] + bounds[3]) / 2.0).abs() < 1e-10);
                assert!(((bounds[1] - bounds[0]) / (bounds[3] - bounds[2]) - aspect).abs() < 1e-10);
                let ctx = Context::default();
                let frame = Rect::from_min_size(
                    egui::pos2(30.0, 40.0),
                    egui::vec2(500.0 * aspect as f32, 500.0),
                );
                let transform = egui_plot::PlotTransform::new(
                    frame,
                    egui_plot::PlotBounds::from_min_max(
                        [bounds[0], bounds[2]],
                        [bounds[1], bounds[3]],
                    ),
                    [false, false],
                );
                let output = ctx.run_ui(egui::RawInput::default(), |ui| {
                    let mut shapes = Vec::new();
                    crate::aoi::preview::Overlay::projected(
                        Arc::clone(&place.area.as_ref().unwrap().preview),
                        projection,
                        1.0,
                    )
                    .shapes(ui, &transform, &mut shapes);
                    for shape in shapes {
                        ui.painter().add(shape);
                    }
                    place.paint_map(ui.painter(), &transform, projection.instance());
                });
                let marker = output
                    .shapes
                    .iter()
                    .find_map(|shape| match &shape.shape {
                        egui::Shape::Circle(circle) if circle.fill == HIGHLIGHT_COLOR => {
                            Some(circle.center)
                        }
                        _ => None,
                    })
                    .expect("projected location marker");
                assert!(marker.distance(frame.center()) < 0.01);
                assert!(output.shapes.iter().any(|shape| matches!(&shape.shape, egui::Shape::Mesh(mesh) if !mesh.vertices.is_empty())));
            }
        }
    }

    #[test]
    fn cmd_and_ctrl_f_open_independent_earth_fields_and_focus_hovered_view() {
        for modifiers in [egui::Modifiers::COMMAND, egui::Modifiers::CTRL] {
            let ctx = Context::default();
            let mut tabs = vec![tab()];
            frame(&ctx, &mut tabs, vec![]);
            frame(
                &ctx,
                &mut tabs,
                vec![
                    egui::Event::PointerMoved(egui::pos2(500.0, 100.0)),
                    key(egui::Key::F, modifiers),
                ],
            );
            frame(&ctx, &mut tabs, vec![egui::Event::Text("Stockholm".into())]);
            assert!(tabs[0].planets.iter().all(|p| p.location_search.open));
            assert_eq!(tabs[0].planets[0].location_search.query, "");
            assert_eq!(tabs[0].planets[1].location_search.query, "Stockholm");
            assert!(tabs[0].planets[1].location_search.pending.is_none());
            frame(
                &ctx,
                &mut tabs,
                vec![key(egui::Key::Escape, egui::Modifiers::NONE)],
            );
            assert!(tabs[0].planets[0].location_search.open);
            assert!(!tabs[0].planets[1].location_search.open);
        }
    }

    #[test]
    fn no_search_shortcut_on_other_planets() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        for planet in &mut tabs[0].planets {
            planet.celestial_body = CelestialBody::Mars;
        }
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::F, egui::Modifiers::COMMAND)],
        );
        assert!(tabs[0].planets.iter().all(|p| !p.location_search.open));
        assert!(ctx.input(|i| i.key_pressed(egui::Key::F)));
    }

    #[test]
    fn editing_query_detaches_old_results() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::F, egui::Modifiers::COMMAND)],
        );
        let old = Mailbox::default();
        tabs[0].planets[0].location_search.pending = Some((Instant::now(), Arc::clone(&old)));
        frame(&ctx, &mut tabs, vec![egui::Event::Text("Paris".into())]);
        *old.lock().unwrap() = Some(parse(FIXTURE));
        frame(&ctx, &mut tabs, vec![]);
        let search = &tabs[0].planets[0].location_search;
        assert_eq!(search.query, "Paris");
        assert!(search.pending.is_none() && search.results.is_empty());
    }

    #[test]
    fn cloned_views_do_not_share_pending_searches() {
        let search = Search {
            open: true,
            query: "Paris".into(),
            pending: Some((Instant::now(), Mailbox::default())),
            ..Default::default()
        };
        let cloned = search.clone();
        assert!(!cloned.open && cloned.pending.is_none() && cloned.query.is_empty());
    }

    #[test]
    fn clicking_a_result_navigates_only_its_view() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        tabs[0].planets[1].location_search = Search {
            open: true,
            submitted: Some("Stockholm".into()),
            results: parse(FIXTURE).unwrap(),
            ..Default::default()
        };
        frame(&ctx, &mut tabs, vec![]);
        let output = frame(&ctx, &mut tabs, vec![]);
        let rect = output
            .shapes
            .iter()
            .find_map(|shape| find_text(&shape.shape, "Stockholm, Sweden"))
            .expect("result is visible");
        for pressed in [true, false] {
            frame(
                &ctx,
                &mut tabs,
                vec![
                    egui::Event::PointerMoved(rect.center()),
                    egui::Event::PointerButton {
                        pos: rect.center(),
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
            );
        }
        assert!(tabs[0].planets[0].location_search.camera.is_none());
        assert!(tabs[0].planets[1].location_search.camera.is_some());
        assert_eq!(
            tabs[0].planets[1].location_search.query,
            "Stockholm, Sweden"
        );
        assert!(tabs[0].planets[1].location_search.results.is_empty());
        assert!(tabs[0].planets[1].location_search.highlight().is_some());
        assert!(tabs[0].planets[0].location_search.highlight().is_none());
    }

    #[test]
    fn hovering_a_result_previews_only_its_marker_without_navigating() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        let label = "Sveavagen 17, Stockholm, Sweden";
        tabs[0].planets[1].location_search = Search {
            open: true,
            submitted: Some("Stockholm".into()),
            results: parse(FIXTURE).unwrap(),
            ..Default::default()
        };
        frame(&ctx, &mut tabs, vec![]);
        let output = frame(&ctx, &mut tabs, vec![]);
        let rect = output
            .shapes
            .iter()
            .find_map(|shape| find_text(&shape.shape, label))
            .unwrap();
        frame(
            &ctx,
            &mut tabs,
            vec![egui::Event::PointerMoved(rect.center())],
        );
        assert!(tabs[0].planets[1]
            .location_search
            .highlight()
            .unwrap()
            .extent
            .is_some());
        assert!(tabs[0].planets[1]
            .location_search
            .highlight()
            .unwrap()
            .area
            .is_none());
        assert!(tabs[0].planets[0].location_search.highlight().is_none());
        assert!(tabs[0]
            .planets
            .iter()
            .all(|p| p.location_search.camera.is_none()));
        frame(
            &ctx,
            &mut tabs,
            vec![egui::Event::PointerMoved(egui::pos2(40.0, 350.0))],
        );
        assert_eq!(tabs[0].planets[1].location_search.active_result, Some(1));
        assert!(tabs[0].planets[0].location_search.highlight().is_none());
    }

    #[test]
    fn selected_area_survives_closing_search_but_new_queries_clear_it() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        tabs[0].planets[0].location_search.open = true;
        tabs[0].planets[0].location_search.select(mapped_place());
        frame(&ctx, &mut tabs, vec![]);
        tabs[0].planets[0].location_search.open = false;
        frame(&ctx, &mut tabs, vec![]);
        assert!(tabs[0].planets[0]
            .location_search
            .highlight()
            .unwrap()
            .extent
            .is_some());
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::F, egui::Modifiers::COMMAND)],
        );
        frame(&ctx, &mut tabs, vec![egui::Event::Text("Paris".into())]);
        assert!(tabs[0].planets[0].location_search.highlight().is_none());
    }

    #[test]
    fn point_only_results_have_a_marker_but_no_invented_area() {
        let place = parse(FIXTURE).unwrap().remove(0);
        let camera = place.camera(1.0);
        let ctx = Context::default();
        let frame = Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(500.0, 500.0));
        let margin = CelestialBody::Earth.radius_km() * 1.15 / camera.zoom;
        let transform = egui_plot::PlotTransform::new(
            frame,
            egui_plot::PlotBounds::from_min_max([-margin, -margin], [margin, margin]),
            [false, false],
        );
        let output = ctx.run_ui(egui::RawInput::default(), |ui| {
            place.paint_globe(
                ui.painter(),
                &transform,
                &camera.rotation,
                CelestialBody::Earth.radius_km(),
                CelestialBody::Earth.flattening(),
            );
        });
        assert!(output
            .shapes
            .iter()
            .any(|shape| matches!(&shape.shape, egui::Shape::Circle(_))));
        assert!(!output
            .shapes
            .iter()
            .any(|shape| matches!(&shape.shape, egui::Shape::Mesh(_))));
    }

    #[test]
    fn late_boundary_results_cannot_replace_a_different_selection() {
        let ctx = Context::default();
        let mut search = Search::default();
        search.select(parse(FIXTURE).unwrap().remove(0));
        let old = Arc::new(Mutex::new(crate::location_boundary::Status::Pending));
        search.boundary = Some(Arc::clone(&old));
        search.boundary_checked = true;
        search.select(parse(FIXTURE).unwrap().remove(1));
        *old.lock().unwrap() = crate::location_boundary::Status::Ready(Ok(mapped_place().area));
        search.poll_boundary(&ctx);
        assert_eq!(
            search.selection.as_ref().unwrap().label,
            "Sveavagen 17, Stockholm, Sweden"
        );
        assert!(search.selection.as_ref().unwrap().area.is_none());
    }

    #[test]
    fn boundary_completion_fits_once_but_does_not_override_manual_navigation() {
        for moved in [false, true] {
            let ctx = Context::default();
            let mut search = Search::default();
            search.select(parse(FIXTURE).unwrap().remove(1));
            search.navigation.take();
            search.boundary_checked = true;
            search.boundary = Some(Arc::new(Mutex::new(
                crate::location_boundary::Status::Ready(Ok(mapped_place().area)),
            )));
            if moved {
                search.user_navigated();
            }
            search.poll_boundary(&ctx);
            assert!(search.selection.as_ref().unwrap().area.is_some());
            assert_eq!(search.navigation.is_none(), moved);
            search.navigation.take();
            search.poll_boundary(&ctx);
            assert!(search.navigation.is_none());
        }
    }

    fn results_in_focused_view(ctx: &Context, tabs: &mut [TabConfig], view: usize) {
        frame(
            ctx,
            tabs,
            vec![
                egui::Event::PointerMoved(egui::pos2(view as f32 * 400.0 + 40.0, 350.0)),
                key(egui::Key::F, egui::Modifiers::COMMAND),
            ],
        );
        let search = &mut tabs[0].planets[view].location_search;
        search.query = "Stockholm".into();
        search.submitted = Some(search.query.clone());
        search.pending = Some((Instant::now(), Arc::new(Mutex::new(Some(parse(FIXTURE))))));
        frame(ctx, tabs, vec![]);
        frame(ctx, tabs, vec![]);
    }

    #[test]
    fn first_result_is_highlighted_and_enter_selects_without_another_request() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        results_in_focused_view(&ctx, &mut tabs, 0);
        assert_eq!(tabs[0].planets[0].location_search.active_result, Some(0));
        assert_eq!(
            tabs[0].planets[0]
                .location_search
                .highlight()
                .unwrap()
                .label,
            "Stockholm, Sweden"
        );
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::Enter, egui::Modifiers::NONE)],
        );
        assert_eq!(
            tabs[0].planets[0]
                .location_search
                .selection
                .as_ref()
                .unwrap()
                .label,
            "Stockholm, Sweden"
        );
        assert!(tabs[0].planets[0].location_search.pending.is_none());
        assert!(tabs[0].planets[0].location_search.camera.is_some());
        assert!(tabs[0].planets[1].location_search.camera.is_none());
    }

    #[test]
    fn first_enter_submits_and_keeps_focus_for_the_second_enter() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        let query = "keyboard-submit-fixture";
        service()
            .lock()
            .unwrap()
            .cache
            .push_back((query.into(), parse(FIXTURE).unwrap()));
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::F, egui::Modifiers::COMMAND)],
        );
        frame(&ctx, &mut tabs, vec![egui::Event::Text(query.into())]);
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::Enter, egui::Modifiers::NONE)],
        );
        assert!(ctx.memory(|memory| memory.has_focus(Id::new(0_usize).with("query"))));
        assert!(tabs[0].planets[0].location_search.pending.is_some());
        let mut release = key(egui::Key::Enter, egui::Modifiers::NONE);
        if let egui::Event::Key { pressed, .. } = &mut release {
            *pressed = false;
        }
        frame(&ctx, &mut tabs, vec![release]);
        assert_eq!(tabs[0].planets[0].location_search.active_result, Some(0));
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::Enter, egui::Modifiers::NONE)],
        );
        assert!(tabs[0].planets[0].location_search.selection.is_some());
        assert!(tabs[0].planets[0].location_search.pending.is_none());
    }

    #[test]
    fn ctrl_j_k_and_tab_cycle_results_without_editing_or_losing_focus() {
        for (next, previous) in [
            (
                (egui::Key::J, egui::Modifiers::CTRL),
                (egui::Key::K, egui::Modifiers::CTRL),
            ),
            (
                (egui::Key::Tab, egui::Modifiers::NONE),
                (egui::Key::Tab, egui::Modifiers::SHIFT),
            ),
            (
                (egui::Key::ArrowDown, egui::Modifiers::NONE),
                (egui::Key::ArrowUp, egui::Modifiers::NONE),
            ),
        ] {
            let ctx = Context::default();
            let mut tabs = vec![tab()];
            results_in_focused_view(&ctx, &mut tabs, 0);
            for (event, expected) in [(next, 1), (next, 0), (previous, 1), (previous, 0)] {
                frame(&ctx, &mut tabs, vec![key(event.0, event.1)]);
                let search = &tabs[0].planets[0].location_search;
                assert_eq!(search.active_result, Some(expected));
                assert_eq!(search.query, "Stockholm");
                assert!(search.selection.is_none() && search.pending.is_none());
                assert!(ctx.memory(|memory| memory.has_focus(Id::new(0_usize).with("query"))));
            }
        }
    }

    #[test]
    fn keyboard_selection_is_scoped_to_the_focused_view() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        results_in_focused_view(&ctx, &mut tabs, 0);
        results_in_focused_view(&ctx, &mut tabs, 1);
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::Tab, egui::Modifiers::NONE)],
        );
        assert_eq!(tabs[0].planets[0].location_search.active_result, Some(0));
        assert_eq!(tabs[0].planets[1].location_search.active_result, Some(1));
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::Enter, egui::Modifiers::NONE)],
        );
        assert!(tabs[0].planets[0].location_search.selection.is_none());
        assert_eq!(
            tabs[0].planets[1]
                .location_search
                .selection
                .as_ref()
                .unwrap()
                .label,
            "Sveavagen 17, Stockholm, Sweden"
        );
    }

    #[test]
    fn stationary_pointer_does_not_undo_keyboard_selection() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        results_in_focused_view(&ctx, &mut tabs, 0);
        let output = frame(&ctx, &mut tabs, vec![]);
        let rect = output
            .shapes
            .iter()
            .find_map(|shape| find_text(&shape.shape, "Stockholm, Sweden"))
            .unwrap();
        frame(
            &ctx,
            &mut tabs,
            vec![egui::Event::PointerMoved(rect.center())],
        );
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::J, egui::Modifiers::CTRL)],
        );
        frame(&ctx, &mut tabs, vec![]);
        assert_eq!(tabs[0].planets[0].location_search.active_result, Some(1));
    }

    #[test]
    fn editing_a_query_resets_result_navigation() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        results_in_focused_view(&ctx, &mut tabs, 0);
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::J, egui::Modifiers::CTRL)],
        );
        frame(&ctx, &mut tabs, vec![egui::Event::Text("a".into())]);
        let search = &tabs[0].planets[0].location_search;
        assert!(
            search.results.is_empty()
                && search.active_result.is_none()
                && search.highlight().is_none()
        );
    }

    #[test]
    fn held_enter_does_not_accidentally_select_new_results() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        let query = "keyboard-repeat-fixture";
        service()
            .lock()
            .unwrap()
            .cache
            .push_back((query.into(), parse(FIXTURE).unwrap()));
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::F, egui::Modifiers::COMMAND)],
        );
        frame(&ctx, &mut tabs, vec![egui::Event::Text(query.into())]);
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::Enter, egui::Modifiers::NONE)],
        );
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::Enter, egui::Modifiers::NONE)],
        );
        assert!(tabs[0].planets[0].location_search.selection.is_none());
        assert_eq!(tabs[0].planets[0].location_search.active_result, Some(0));
    }

    #[test]
    fn provider_links_are_in_info_content_and_not_the_search_panel() {
        let ctx = Context::default();
        let output = ctx.run_ui(egui::RawInput::default(), |ui| sources_ui(ui));
        for label in ["Photon", "Nominatim", "OpenStreetMap contributors"] {
            assert!(output
                .shapes
                .iter()
                .any(|shape| find_text(&shape.shape, label).is_some()));
        }
        let mut tabs = vec![tab()];
        results_in_focused_view(&ctx, &mut tabs, 0);
        let output = frame(&ctx, &mut tabs, vec![]);
        for label in ["Photon", "Nominatim", "OpenStreetMap contributors"] {
            assert!(!output
                .shapes
                .iter()
                .any(|shape| find_text(&shape.shape, label).is_some()));
        }
    }

    #[test]
    fn keyboard_navigation_scrolls_long_result_rows_into_view() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        results_in_focused_view(&ctx, &mut tabs, 0);
        tabs[0].planets[0].location_search.results = (0..RESULT_LIMIT)
            .map(|index| {
                let mut place = parse(FIXTURE).unwrap().remove(0);
                place.label = format!(
                    "Place {index}, {}",
                    "Long administrative district name and municipality, ".repeat(4)
                );
                place
            })
            .collect();
        frame(&ctx, &mut tabs, vec![]);
        let last = tabs[0].planets[0]
            .location_search
            .results
            .last()
            .unwrap()
            .label
            .clone();
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::K, egui::Modifiers::CTRL)],
        );
        for _ in 0..60 {
            frame(&ctx, &mut tabs, vec![]);
        }
        let output = frame(&ctx, &mut tabs, vec![]);
        assert_eq!(
            tabs[0].planets[0].location_search.active_result,
            Some(RESULT_LIMIT - 1)
        );
        assert!(output
            .shapes
            .iter()
            .any(|shape| find_text(&shape.shape, &last)
                .is_some_and(|rect| shape.clip_rect.contains(rect.center()))));
    }

    #[test]
    fn enter_commits_ime_text_without_submitting_or_selecting() {
        let ctx = Context::default();
        let mut tabs = vec![tab()];
        results_in_focused_view(&ctx, &mut tabs, 0);
        frame(
            &ctx,
            &mut tabs,
            vec![
                egui::Event::Ime(egui::ImeEvent::Enabled),
                egui::Event::Ime(egui::ImeEvent::Preedit("東京".into())),
            ],
        );
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::Enter, egui::Modifiers::NONE)],
        );
        assert!(tabs[0].planets[0].location_search.pending.is_none());
        assert!(tabs[0].planets[0].location_search.selection.is_none());
        let mut release = key(egui::Key::Enter, egui::Modifiers::NONE);
        if let egui::Event::Key { pressed, .. } = &mut release {
            *pressed = false;
        }
        frame(
            &ctx,
            &mut tabs,
            vec![
                egui::Event::Ime(egui::ImeEvent::Commit("東京".into())),
                release,
            ],
        );
        assert!(!tabs[0].planets[0].location_search.ime_composing);
        assert!(tabs[0].planets[0].location_search.pending.is_none());
        assert!(tabs[0].planets[0].location_search.selection.is_none());
        tabs[0].planets[0].location_search.results = parse(FIXTURE).unwrap();
        frame(
            &ctx,
            &mut tabs,
            vec![key(egui::Key::Enter, egui::Modifiers::NONE)],
        );
        assert!(tabs[0].planets[0].location_search.selection.is_some());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "calls the public Photon API once"]
    fn public_api_returns_places_and_allows_browser_requests() {
        let mut request = ehttp::Request::get(search_url("Stockholm"));
        request.headers.insert("Origin", "http://localhost:8080");
        let response = ehttp::fetch_blocking(&request).unwrap();
        assert!(response.ok, "HTTP {}", response.status);
        assert_eq!(
            response.headers.get("access-control-allow-origin"),
            Some("*")
        );
        let places = parse(&response.bytes).unwrap();
        assert!(places
            .iter()
            .any(|p| p.label.contains("Sweden") && (p.lat - 59.3).abs() < 0.1));
    }
}
