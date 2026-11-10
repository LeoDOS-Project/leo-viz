use super::*;
use crate::config::{ConstellationConfig, TabSettings};

const RADIUS: f64 = 6371.0;
const ID: &str = "globe-click-priority";

#[derive(Default)]
struct Selection {
    pending: Vec<SatelliteCamera>,
    cameras: Vec<SatelliteCamera>,
    removed: Vec<usize>,
    counter: usize,
    jobs: Vec<crate::aoi::preview::Hit>,
    label: Option<(bool, usize, egui::Pos2)>,
    observation: Option<(f64, f64)>,
    dragging: Option<(usize, usize, bool, usize)>,
    context: Option<(egui::Pos2, f64, f64)>,
    pinned: HashSet<crate::config::PinnedIsl>,
    areas: Vec<AreaOfInterest>,
}

fn constellation() -> DrawConstellationData {
    let mut config = ConstellationConfig::new(0);
    config.sats_per_plane = 1;
    config.num_planes = 1;
    let orbit = config.constellation(RADIUS, 398600.4418, 0.0, RADIUS);
    let mut positions = orbit.satellite_positions(0.0);
    positions[0].x = 0.0;
    positions[0].y = 0.0;
    positions[0].z = RADIUS + config.altitude_km;
    (orbit, positions, 0, 0, 0, "Test".into())
}

fn area() -> Arc<crate::aoi::preview::Preview> {
    Arc::new(
        crate::aoi::preview::Preview::from_geojson(
            serde_json::json!({
                "type":"Polygon", "coordinates":[[[-95,-5],[-85,-5],[-85,5],[-95,5],[-95,-5]]]
            }),
            [-90.0, 0.0],
            [70, 220, 240],
        )
        .unwrap(),
    )
}

fn flags() -> View3DFlags {
    let s = TabSettings::default();
    View3DFlags {
        show_orbits: false,
        show_axes: false,
        show_magnetic_axis: false,
        show_coverage: false,
        show_links: false,
        show_gs_links: false,
        hide_behind_earth: true,
        single_color: true,
        dark_mode: false,
        show_routing_paths: false,
        show_proxy_links: false,
        show_path_info: false,
        show_link_info: false,
        show_manhattan_path: false,
        show_shortest_path: false,
        show_radiation_path: false,
        radiation_weight: s.radiation_weight,
        routing_width: s.routing_width,
        routing_node_scale: s.routing_node_scale,
        show_asc_desc_colors: false,
        color_ascending: s.color_ascending,
        color_descending: s.color_descending,
        color_links: s.color_links,
        show_sat_labels: false,
        show_altitude_lines: false,
        altitude_line_width: s.altitude_line_width,
        show_inclination_bounds: false,
        render_planet: false,
        fixed_sizes: true,
        show_sat_border: false,
        show_polar_circle: false,
        show_equator: false,
        show_graticule: false,
        show_crosshairs: false,
        show_terminator: false,
        show_eclipse: false,
        show_sun: false,
        earth_fixed_camera: true,
        show_clouds: false,
        show_day_night: false,
        show_city_lights: false,
        show_stars: false,
        show_borders: false,
        show_cities: false,
        trackpad_rotate: false,
        north_up: false,
        enabled_moons: HashSet::new(),
        moon_inclination_override: None,
        show_moon_orbits: false,
        show_moon_lines: false,
        show_moon_labels: false,
        moon_camera_distance_km: s.moon_camera_distance_km,
        tle_monochrome: false,
        show_ground_tracks: false,
    }
}

fn frame(
    ctx: &egui::Context,
    time: f64,
    events: Vec<egui::Event>,
    data: &[DrawConstellationData],
    preview: &Arc<crate::aoi::preview::Preview>,
    state: &mut Selection,
    sampler: bool,
) -> egui_plot::PlotTransform {
    let mut transform = None;
    let _ = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(600.0, 600.0),
            )),
            time: Some(time),
            events,
            ..Default::default()
        },
        |ui| {
            draw_3d_view(
                ui,
                Draw3dInput {
                    search_highlight: None,
                    selected_aoi: None,
                    observation_sampler: sampler,
                    observation_marker: None,
                    pushbroom: None,
                    generated_aois: vec![(preview, 1.0, 0.1)],
                    id: ID,
                    constellations: data,
                    flags: flags(),
                    coverage_angle: 45.0,
                    rotation: Matrix3::identity(),
                    satellite_rotation: Matrix3::identity(),
                    width: 500.0,
                    height: 500.0,
                    earth_texture: None,
                    zoom: 1.0,
                    sat_radius: 3.0,
                    link_width: 1.0,
                    planet_radius: RADIUS,
                    flattening: 0.0,
                    gpu_available: false,
                    body_key: (CelestialBody::Earth, Skin::Default, TextureResolution::R512),
                    body_rotation: &Matrix3::identity(),
                    sun_dir: [1.0, 0.0, 0.0],
                    time: 0.0,
                    ground_stations_locked: false,
                    device_layers: &[],
                    body_rot_angle: 0.0,
                    drag_tab_planet: (0, 0),
                    detail_bounds: None,
                    geo_borders: &[],
                    geo_cities: &[],
                    conjunction_lines: &[],
                    conjunction_heatmap: &[],
                    correcting_sats: &HashSet::new(),
                    hit_sats: &HashSet::new(),
                    radiation: None,
                    moon_handles: &HashMap::new(),
                    physics_colors: &HashMap::new(),
                    physics_info: &HashMap::new(),
                    ground_tracks: &[],
                    flash_intensities: &[],
                },
                Draw3dState {
                    job_pick: &mut state.jobs,
                    observation_pick: &mut state.observation,
                    pending_cameras: &mut state.pending,
                    camera_id_counter: &mut state.counter,
                    satellite_cameras: &mut state.cameras,
                    cameras_to_remove: &mut state.removed,
                    pinned_isls: &mut state.pinned,
                    ground_stations: &mut [],
                    areas_of_interest: &mut state.areas,
                    dragging_place: &mut state.dragging,
                    context_menu_request: &mut state.context,
                    label_click_request: &mut state.label,
                },
            );
            transform = Some(
                egui_plot::PlotMemory::load(ctx, ui.make_persistent_id(egui::Id::new(ID)))
                    .unwrap()
                    .transform(),
            );
        },
    );
    transform.unwrap()
}

fn click(
    ctx: &egui::Context,
    time: f64,
    pos: egui::Pos2,
    data: &[DrawConstellationData],
    preview: &Arc<crate::aoi::preview::Preview>,
    state: &mut Selection,
    sampler: bool,
) {
    frame(
        ctx,
        time,
        vec![egui::Event::PointerMoved(pos)],
        data,
        preview,
        state,
        sampler,
    );
    for (offset, pressed) in [(0.05, true), (0.1, false)] {
        frame(
            ctx,
            time + offset,
            vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            }],
            data,
            preview,
            state,
            sampler,
        );
    }
}

#[test]
fn clicking_a_satellite_over_an_aoi_opens_and_closes_its_camera() {
    let ctx = egui::Context::default();
    let data = vec![constellation()];
    let preview = area();
    let mut state = Selection::default();
    frame(&ctx, 0.0, vec![], &data, &preview, &mut state, false);
    let transform = frame(&ctx, 0.1, vec![], &data, &preview, &mut state, false);
    let pos = transform.position_from_point(&PlotPoint::new(0.0, 0.0));
    assert!(!crate::aoi::preview::Overlay::new(
        Arc::clone(&preview),
        Matrix3::identity(),
        RADIUS,
        0.0,
        1.0
    )
    .pick(pos, &transform)
    .is_empty());
    click(&ctx, 1.0, pos, &data, &preview, &mut state, false);
    assert_eq!(state.pending.len(), 1);
    assert!(state.jobs.is_empty());
    assert!(state.label.is_none());
    let camera = state.pending.pop().unwrap();
    assert_eq!(camera.label, "P0:S0");
    let id = camera.id;
    state.cameras.push(camera);
    click(&ctx, 2.0, pos, &data, &preview, &mut state, false);
    assert_eq!(state.removed, vec![id]);
    assert!(state.jobs.is_empty());
}

#[test]
fn aoi_clicks_outside_the_satellite_still_select_the_aoi() {
    let ctx = egui::Context::default();
    let data = vec![constellation()];
    let preview = area();
    let mut state = Selection::default();
    frame(&ctx, 0.0, vec![], &data, &preview, &mut state, false);
    let transform = frame(&ctx, 0.1, vec![], &data, &preview, &mut state, false);
    let pos = transform.position_from_point(&PlotPoint::new(300.0, 0.0));
    click(&ctx, 1.0, pos, &data, &preview, &mut state, false);
    assert_eq!(state.jobs.len(), 1);
    assert!(state.pending.is_empty());
}

#[test]
fn a_surface_label_cannot_steal_a_highlighted_satellite_click() {
    for satellite in [false, true] {
        let ctx = egui::Context::default();
        let data = if satellite {
            vec![constellation()]
        } else {
            Vec::new()
        };
        let preview = area();
        let mut state = Selection::default();
        state.areas.push(AreaOfInterest {
            name: "Test area".into(),
            lat: 0.0,
            lon: -90.0,
            radius_km: 100.0,
            color: egui::Color32::RED,
            ground_station_idx: None,
            job_mode: AoiJobMode::Route,
            job_n: 3,
            reducer_placement: crate::config::SpaceCompReducerPlacement::NearMappers,
            show_spacecomp_flow: false,
            selected: false,
        });
        frame(&ctx, 0.0, vec![], &data, &preview, &mut state, false);
        let transform = frame(&ctx, 0.1, vec![], &data, &preview, &mut state, false);
        let pos = transform.position_from_point(&PlotPoint::new(0.0, 0.0)) - egui::vec2(0.0, 3.0);
        click(&ctx, 1.0, pos, &data, &preview, &mut state, false);
        if satellite {
            assert_eq!(state.pending.len(), 1);
            assert!(state.label.is_none());
        } else {
            assert!(matches!(state.label, Some((false, 0, _))));
        }
        assert!(state.jobs.is_empty());
    }
}

#[test]
fn observation_sampling_still_takes_priority_over_satellites_and_aois() {
    let ctx = egui::Context::default();
    let data = vec![constellation()];
    let preview = area();
    let mut state = Selection::default();
    frame(&ctx, 0.0, vec![], &data, &preview, &mut state, true);
    let transform = frame(&ctx, 0.1, vec![], &data, &preview, &mut state, true);
    let pos = transform.position_from_point(&PlotPoint::new(0.0, 0.0));
    click(&ctx, 1.0, pos, &data, &preview, &mut state, true);
    assert!(state.observation.is_some());
    assert!(state.jobs.is_empty() && state.pending.is_empty());
}

#[test]
fn picking_uses_the_nearest_satellite_and_honors_earth_occlusion() {
    let mut data = vec![constellation()];
    data[0].1[0].x = 10.0;
    let mut other = constellation();
    other.1[0].x = 2.0;
    data.push(other);
    let pick = |data: &[DrawConstellationData], hide| {
        pick_globe_satellite(
            data,
            &Matrix3::identity(),
            PlotPoint::new(0.0, 0.0),
            20.0,
            RADIUS,
            hide,
        )
    };
    assert_eq!(pick(&data, true), Some((1, 0)));
    data[1].1[0].z *= -1.0;
    assert_eq!(pick(&data, true), Some((0, 0)));
    assert_eq!(pick(&data, false), Some((1, 0)));
}
