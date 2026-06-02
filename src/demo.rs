//! Demo mode setup with pre-configured multi-tab constellation layouts.

use crate::celestial::CelestialBody;
use crate::config::{
    AoiJobMode, AreaOfInterest, ConstellationConfig, GroundStation, PlanetConfig, Preset, TabConfig,
};
use crate::tle::{TleLoadState, TlePreset};
use crate::walker::WalkerType;
use crate::{App, ViewerState};
use egui_dock::DockState;

pub fn inclination_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Inclination: 90° vs 60°".to_string());
    tab.title = "Inclination: 90° vs 60°".to_string();
    tab.description = indoc::indoc! {"
            Two constellations with different inclinations (90° polar vs 60° inclined).

            **Inclination** is the angle between the orbital planes and the planet's equator.

            A 90° inclination means the satellites pass over the poles, while a 60° inclination means they reach up to 60° latitude north and south.
        "}.to_string();
    tab.settings.sat_radius = 4.0;
    tab.settings.link_width = 1.0;
    tab.settings.show_links = false;
    tab.settings.auto_rotate = true;
    tab.settings.auto_rotate_speed = 3.0;
    for (inc, label) in [(90.0, "90° inclination"), (60.0, "60° inclination")] {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(format!("Earth ({})", label));
        planet.celestial_body = CelestialBody::Earth;
        let mut cons = ConstellationConfig::new(0);
        cons.sats_per_plane = 20;
        cons.num_planes = 20;
        cons.inclination = inc;
        cons.altitude_km = 780.0;
        cons.walker_type = WalkerType::Star;
        planet.constellations.push(cons);
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn walker_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Walker: Star vs Delta".to_string());
    tab.title = "Walker: Star vs Delta".to_string();
    tab.description = indoc::indoc! {"
            Two constellations with different Walker types (Star vs Delta).

            The **Walker type** determines how orbital planes are distributed around the planet.

            • **Star** constellations optimise for polar coverage by distributing orbital planes over 180° of longitude, which means satellites ascend on one side of the planet and descend on the opposite side.
            • **Delta** constellations optimise for mid-latitude coverage by distributing orbital planes over 360° of longitude, which means satellites ascend and descend on both sides of the planet.
        "}.to_string();
    tab.settings.sat_radius = 3.0;
    tab.settings.link_width = 1.0;
    tab.settings.show_links = false;
    tab.settings.earth_fixed_camera = true;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(75.0_f64.to_radians(), 0.0);
    for (wt, label) in [
        (WalkerType::Star, "Walker Star"),
        (WalkerType::Delta, "Walker Delta"),
    ] {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(format!("Earth ({})", label));
        planet.celestial_body = CelestialBody::Earth;
        let mut cons = ConstellationConfig::new(0);
        cons.sats_per_plane = 20;
        cons.num_planes = 20;
        cons.inclination = 70.0;
        cons.altitude_km = 500.0;
        cons.walker_type = wt;
        planet.constellations.push(cons);
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn coverage_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Coverage: 300km vs 600km".to_string());
    tab.title = "Coverage: 300km vs 600km".to_string();
    tab.description = indoc::indoc! {"
            Two identical Walker Delta constellations with a 50° half-cone coverage angle, at different altitudes (300 km vs 600 km).

            The **coverage angle** is a property of the satellite's antenna beamwidth and is fixed, but the ground footprint grows with altitude; higher satellites see a larger area, so fewer of them are needed for the same global coverage.
        "}.to_string();
    tab.settings.sat_radius = 3.0;
    tab.settings.link_width = 1.0;
    tab.settings.show_links = false;
    tab.settings.show_coverage = true;
    tab.settings.coverage_angle = 100.0;
    tab.settings.earth_fixed_camera = true;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(45.0_f64.to_radians(), 0.0);
    for (alt, label) in [(300.0, "300 km alt"), (600.0, "600 km alt")] {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(format!("Earth ({}, 50° half-cone)", label));
        planet.celestial_body = CelestialBody::Earth;
        let mut cons = ConstellationConfig::new(0);
        cons.sats_per_plane = 20;
        cons.num_planes = 20;
        cons.inclination = 53.0;
        cons.altitude_km = alt;
        cons.walker_type = WalkerType::Delta;
        planet.constellations.push(cons);
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn phasing_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Phasing: F=0 vs F=0.5".to_string());
    tab.title = "Phasing: F=0 vs F=0.5".to_string();
    tab.description = indoc::indoc! {"
            Two constellations with different phasing (F=0 vs F=0.5).

            **Phasing** determines the relative positions of satellites in adjacent planes.

            • **F=0** means satellites in adjacent planes are aligned (e.g., all ascending nodes line up).
            • **F=0.5** means satellites in adjacent planes are offset by half the inter-satellite spacing, creating a more staggered pattern.

            Phasing is primarily used to prevent collisions between satellites in different orbits.
        "}.to_string();
    tab.settings.sat_radius = 3.0;
    tab.settings.link_width = 1.0;
    tab.settings.show_links = false;
    tab.settings.earth_fixed_camera = true;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(90.0_f64.to_radians(), 0.0);
    tab.settings.zoom = 2.5;
    tab.settings.show_sat_border = true;
    for (f, label) in [(0.0, "F=0 phasing"), (0.5, "F=0.5 phasing")] {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(format!("Earth ({})", label));
        planet.celestial_body = CelestialBody::Earth;
        let mut cons = ConstellationConfig::new(0);
        cons.sats_per_plane = 10;
        cons.num_planes = 10;
        cons.inclination = 90.0;
        cons.altitude_km = 5000.0;
        cons.walker_type = WalkerType::Star;
        cons.phasing = f;
        planet.constellations.push(cons);
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn altitude_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Altitude: VLEO/LEO/MEO/GEO".to_string());
    tab.title = "Altitude: VLEO/LEO/MEO/GEO".to_string();
    tab.description = indoc::indoc! {"
            Four orbital shells at different altitudes (300 km VLEO, 2000 km LEO, 20000 km MEO, 35786 km GEO).

            **Altitude** is the height of the satellites' orbits above the planet's surface.

            Lower-altitude satellites move faster, since orbital speed follows **v = √(μ/r)**, where μ is the planet's gravitational parameter and r is the orbital radius. Halve the radius and the speed grows by √2; quadruple it and the speed halves.

            • **VLEO** (Very Low Earth Orbit) refers to altitudes below 500 km, where atmospheric drag is significant — satellites need thrusters firing periodically (electric propulsion is common) just to stay aloft, otherwise they reenter within months.
            • **LEO** (Low Earth Orbit) refers to altitudes between 500 km and 2000 km, where many Earth observation and communication satellites operate.
            • **MEO** (Medium Earth Orbit) refers to altitudes between 2000 km and ~35000 km, where navigation constellations like GPS operate.
            • **GEO** (Geostationary Earth Orbit) is at approximately 35786 km altitude, where satellites appear stationary relative to the Earth's surface. Real GEO satellites always orbit directly above the equator (i≈0°); an inclined orbit at GEO altitude traces a figure-eight ground track rather than staying fixed.

            For reference, the **Moon** orbits at about 384,400 km (over 10× farther than GEO).
        "}.to_string();
    tab.settings.sat_radius = 3.0;
    tab.settings.link_width = 2.0;
    tab.settings.show_links = false;
    tab.settings.earth_fixed_camera = false;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(90.0_f64.to_radians(), 0.0);
    tab.settings.zoom = 10000.0 / 60000.0;
    tab.settings.show_sat_labels = true;
    tab.settings.show_altitude_lines = true;
    tab.settings.altitude_line_width = 2.0;
    tab.settings.speed = 2000.0;
    tab.settings.fixed_sizes = true;
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    let altitudes = [
        (300.0, 0, "VLEO"),
        (2000.0, 1, "LEO"),
        (20000.0, 2, "MEO"),
        (35786.0, 3, "GEO"),
    ];
    for (idx, (alt, color, name)) in altitudes.iter().enumerate() {
        let mut cons = ConstellationConfig::new(*color);
        cons.sats_per_plane = 1;
        cons.num_planes = 1;
        cons.inclination = 0.0;
        cons.altitude_km = *alt;
        cons.label = Some(name.to_string());
        planet.constellations.push(cons);
        v.camera_id_counter += 1;
        planet
            .satellite_cameras
            .push(crate::config::SatelliteCamera {
                id: v.camera_id_counter,
                label: name.to_string(),
                constellation_idx: idx,
                plane: 0,
                sat_index: 0,
                screen_pos: None,
            });
    }
    tab.planets.push(planet);
    v.tabs.push(tab);
}

fn ground_track_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Ground Tracks".to_string());
    tab.title = "Ground Tracks".to_string();
    tab.description = indoc::indoc! {"
            Three satellites at different inclinations, each drawing the path of its sub-satellite point on the Earth's surface.

            The **ground track** is the curve traced by the point directly below the satellite as it orbits. On a rotating Earth, the track shifts west on every successive orbit because the planet has turned under the satellite.

            • A satellite at inclination **i** traces a wave between latitudes **+i** and **−i**.
            • Earth rotates ~22.5° during a typical LEO orbit (~90 min), so each pass lands ~22.5° west of the previous one.
            • After enough orbits, the track fills in a swirl pattern across every longitude.

            Certain altitudes produce **repeat cycles**; after N orbits the ground track re-aligns exactly, useful for imaging missions that need to revisit the same spot (e.g., Landsat's 16-day, 233-orbit cycle).
        "}.to_string();
    tab.settings.sat_radius = 3.5;
    tab.settings.link_width = 1.0;
    tab.settings.show_links = false;
    tab.settings.show_orbits = false;
    tab.settings.show_ground_tracks = true;
    tab.settings.earth_fixed_camera = true;
    tab.settings.speed = 1000.0;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(0.0, 0.0);
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    for (idx, (inc, label, color)) in [
        (30.0_f64, "i=30°", 0),
        (53.0_f64, "i=53°", 1),
        (80.0_f64, "i=80°", 2),
    ]
    .iter()
    .enumerate()
    {
        let mut cons = ConstellationConfig::new(*color);
        cons.sats_per_plane = 1;
        cons.num_planes = 1;
        cons.inclination = *inc;
        cons.altitude_km = 700.0;
        cons.walker_type = WalkerType::Delta;
        cons.label = Some(label.to_string());
        planet.constellations.push(cons);
        v.camera_id_counter += 1;
        planet
            .satellite_cameras
            .push(crate::config::SatelliteCamera {
                id: v.camera_id_counter,
                label: label.to_string(),
                constellation_idx: idx,
                plane: 0,
                sat_index: 0,
                screen_pos: None,
            });
    }
    tab.planets.push(planet);
    v.tabs.push(tab);
}

fn eccentricity_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Eccentricity: ω=0 vs ω=90".to_string());
    tab.title = "Eccentricity: ω=0 vs ω=90".to_string();
    tab.description = indoc::indoc! {"
            Two constellations with the same eccentricity but different arguments of periapsis (ω=0° vs ω=90°).

            **Eccentricity** describes how elongated an orbit is; 0 is circular, values closer to 1 are progressively more elliptical.

            • **Periapsis** is the point in the orbit closest to the planet, where the satellite moves fastest.
            • **Apoapsis** is the point furthest from the planet, where the satellite moves slowest.
            • **Argument of periapsis (ω)** is the angle from the ascending node to periapsis, measured in the orbital plane. It rotates the ellipse within the plane, changing which latitude the satellite lingers over.

            Highly eccentric orbits like **Molniya** (ω≈270°, e≈0.74) exploit this to dwell over high northern latitudes for hours per revolution.
        "}.to_string();
    tab.settings.sat_radius = 6.0;
    tab.settings.link_width = 2.0;
    tab.settings.show_links = false;
    tab.settings.show_orbits = true;
    tab.settings.earth_fixed_camera = false;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(30.0_f64.to_radians(), 0.0);
    tab.settings.zoom = 10000.0 / 30000.0;
    tab.settings.speed = 250.0;
    for (omega, label) in [(0.0, "ω=0° periapsis"), (90.0, "ω=90° periapsis")] {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(format!("Earth ({})", label));
        planet.celestial_body = CelestialBody::Earth;
        let mut cons = ConstellationConfig::new(0);
        cons.sats_per_plane = 30;
        cons.num_planes = 30;
        cons.inclination = 63.4;
        cons.altitude_km = 500.0;
        cons.eccentricity = 0.5;
        cons.arg_periapsis = omega;
        cons.label = Some(label.to_string());
        planet.constellations.push(cons);
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn sso_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Sun-Synchronous Orbit".to_string());
    tab.title = "Sun-Synchronous Orbit".to_string();
    tab.description = indoc::indoc! {"
            Two orbital planes at the same altitude (600 km) viewed from a sun-fixed camera: one sun-synchronous (i≈97.8°), one at i=60°.

            A **sun-synchronous orbit (SSO)** is one whose orbital plane rotates around Earth at the same rate the Sun appears to move across the sky (~0.9856°/day). It therefore maintains a fixed orientation relative to the Sun throughout the year.

            • The plane's **RAAN** (the longitude where the orbit crosses the equator going north) precesses at exactly the required rate thanks to the torque from Earth's equatorial bulge (the **J2** perturbation).
            • This only works at a specific **inclination** for each altitude; typically slightly retrograde (~97–98° in LEO).

            In the sun-fixed view, the SSO plane stays locked to the terminator while the non-SSO plane visibly drifts across it. SSO is the orbital regime used by most Earth-observation satellites so every image is taken at the same local solar time.
        "}.to_string();
    tab.settings.sat_radius = 2.5;
    tab.settings.link_width = 3.0;
    tab.settings.show_links = false;
    tab.settings.show_orbits = true;
    tab.settings.show_terminator = true;
    tab.settings.show_day_night = true;
    tab.settings.speed = 100000.0;
    tab.settings.sun_fixed_camera = true;
    tab.settings.hide_behind_earth = false;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(45.0_f64.to_radians(), 0.0);
    tab.settings.zoom = 10000.0 / 10000.0;
    tab.settings.reset_time_on_switch = true;
    // SSO plane (inclination auto = ~97.8°, J2 precesses RAAN ~1°/day eastward,
    // matching Earth's motion around the Sun → plane stays locked to terminator).
    {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new("SSO (plane locked to Sun)".to_string());
        planet.celestial_body = CelestialBody::Earth;
        let mut cons = ConstellationConfig::new(0);
        cons.sats_per_plane = 12;
        cons.num_planes = 1;
        cons.altitude_km = 600.0;
        cons.sso = true;
        cons.raan_offset = 55.0;
        cons.walker_type = WalkerType::Star;
        cons.propagator = crate::config::Propagator::J2;
        cons.label = Some("SSO".to_string());
        if let Some(inc) = ConstellationConfig::sso_inclination(
            cons.altitude_km,
            cons.eccentricity,
            CelestialBody::Earth.mu(),
            CelestialBody::Earth.j2(),
            CelestialBody::Earth.radius_km(),
            CelestialBody::Earth.equatorial_radius_km(),
            // Match the simulation's Sun RA rate (uses DAYS_PER_YEAR = 365.0)
            // rather than Earth's actual 365.25-day sidereal year.
            crate::time::DAYS_PER_YEAR,
        ) {
            cons.inclination = inc;
        }
        planet.constellations.push(cons);
        tab.planets.push(planet);
    }
    // Non-SSO plane (inclination 60°, J2 precesses RAAN at a rate that does NOT
    // match Earth's motion around the Sun → plane drifts relative to terminator).
    {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new("Non-SSO (plane drifts)".to_string());
        planet.celestial_body = CelestialBody::Earth;
        let mut cons = ConstellationConfig::new(7);
        cons.sats_per_plane = 12;
        cons.num_planes = 1;
        cons.altitude_km = 600.0;
        cons.sso = false;
        cons.inclination = 60.0;
        cons.raan_offset = 90.0;
        cons.walker_type = WalkerType::Delta;
        cons.propagator = crate::config::Propagator::J2;
        cons.label = Some("i=60°".to_string());
        planet.constellations.push(cons);
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn isl_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("ISL Topology".to_string());
    tab.title = "ISL Topology".to_string();
    tab.description = indoc::indoc! {"
            The same constellation rendered with two different inter-satellite link topologies (4-ISL vs 8-ISL).

            An **inter-satellite link (ISL)** is a direct communication link between neighbouring satellites, typically laser-based. ISLs let data be routed satellite-to-satellite without touching a ground station.

            • **4-ISL** (Manhattan grid) gives each satellite links to its in-plane fore/aft and cross-plane left/right neighbours.
            • **8-ISL** adds four diagonal links, giving shorter-hop routes across the grid at the cost of more terminals per satellite.
        "}.to_string();
    tab.settings.sat_radius = 3.0;
    tab.settings.link_width = 0.25;
    tab.settings.show_links = true;
    tab.settings.show_orbits = false;
    tab.settings.earth_fixed_camera = true;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(60.0_f64.to_radians(), 0.0);
    tab.settings.auto_rotate = true;
    tab.settings.auto_rotate_speed = 3.0;
    tab.settings.auto_rotate_axis_lat = 0.0;
    tab.settings.auto_rotate_axis_lon = 0.0;
    for (isl, label) in [(4, "4 ISLs per sat"), (8, "8 ISLs per sat")] {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(format!("Earth ({})", label));
        planet.celestial_body = CelestialBody::Earth;
        let mut cons = ConstellationConfig::new(0);
        cons.sats_per_plane = 40;
        cons.num_planes = 20;
        cons.inclination = 86.4;
        cons.altitude_km = 780.0;
        cons.walker_type = WalkerType::Star;
        cons.isl_neighbors = isl;
        planet.constellations.push(cons);
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn torus_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Torus View".to_string());
    tab.title = "Torus View".to_string();
    tab.description = indoc::indoc! {"
            The constellation's ISL topology unwrapped onto a torus.

            A **torus** is the natural topological shape of a Walker constellation: orbital planes wrap around one axis of the donut, and satellites within each plane wrap around the other.

            Mapping the network onto a torus makes the routing structure explicit; what looks like a tangle of links on a sphere becomes a regular 2D grid wrapping in both directions.
        "}.to_string();
    tab.settings.show_torus = true;
    tab.settings.sat_radius = 3.0;
    tab.settings.show_links = true;
    tab.settings.show_orbits = false;
    tab.settings.earth_fixed_camera = true;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(65.0_f64.to_radians(), 0.0);
    tab.settings.camera_roll = 45.0;
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    let mut cons = ConstellationConfig::new(0);
    cons.sats_per_plane = 40;
    cons.num_planes = 40;
    cons.inclination = 80.0;
    cons.altitude_km = 780.0;
    cons.walker_type = WalkerType::Delta;
    cons.isl_neighbors = 4;
    planet.constellations.push(cons);
    tab.planets.push(planet);
    v.tabs.push(tab);
}

fn routing_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("ISL Routing".to_string());
    tab.title = "ISL Routing".to_string();
    tab.description = indoc::indoc! {"
            Two satellites (Src and Dst) with two different routing paths drawn between them.

            **Routing** is the choice of which ISL hops a packet takes from source to destination.

            • **Manhattan** (shown in **red**) routing walks along the constellation grid in two phases (cross-plane then in-plane). It's simple and fast to compute, but may take more hops than necessary.
            • **Shortest-path** (shown in **green**) routing (Dijkstra) finds the fewest-hop route through the link graph. It's optimal but costlier to compute.

            Satellites are coloured by whether they are currently ascending or descending, which matters because grid topology breaks at the seams where adjacent planes are counter-rotating.
        "}.to_string();
    tab.settings.sat_radius = 4.0;
    tab.settings.link_width = 1.0;
    tab.settings.show_links = true;
    tab.settings.show_orbits = false;
    tab.settings.show_routing_paths = true;
    tab.settings.show_manhattan_path = true;
    tab.settings.show_shortest_path = true;
    tab.settings.routing_width = 3.0;
    tab.settings.routing_node_scale = 1.0;
    tab.settings.show_asc_desc_colors = true;
    tab.settings.color_links = egui::Color32::from_rgb(100, 100, 100);
    tab.settings.camera_mode = crate::config::CameraMode::TrackSatellite;
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    let mut cons = ConstellationConfig::new(0);
    cons.sats_per_plane = 30;
    cons.num_planes = 30;
    cons.inclination = 60.0;
    cons.altitude_km = 780.0;
    cons.walker_type = WalkerType::Delta;
    cons.phasing = 1.0;
    cons.isl_neighbors = 4;
    planet.constellations.push(cons);
    v.camera_id_counter += 1;
    planet
        .satellite_cameras
        .push(crate::config::SatelliteCamera {
            id: v.camera_id_counter,
            label: "Src".to_string(),
            constellation_idx: 0,
            plane: 0,
            sat_index: 0,
            screen_pos: None,
        });
    v.camera_id_counter += 1;
    planet
        .satellite_cameras
        .push(crate::config::SatelliteCamera {
            id: v.camera_id_counter,
            label: "Dst".to_string(),
            constellation_idx: 0,
            plane: 3,
            sat_index: 3,
            screen_pos: None,
        });
    tab.planets.push(planet);
    v.tabs.push(tab);
}

fn kessler_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Kessler Syndrome".to_string());
    tab.title = "Kessler Syndrome".to_string();
    tab.description = indoc::indoc! {"
            Two Earths each containing two crossing constellations; on the left at overlapping altitudes, on the right cleanly separated by 500 km.

            **Kessler syndrome** is a runaway cascade where each collision between orbiting objects produces debris fragments which then collide with other objects, producing more debris, in a self-sustaining chain reaction.

            • On the **crossing** planet, the two constellations share the same altitude shell, so inter-plane crossings can collide; each collision spawns 15 fragments (red X) that may trigger further collisions.
            • On the **separated** planet, a 500 km altitude gap keeps the two shells from ever meeting, so no collisions occur.

            Phasing is enabled within each constellation to prevent intra-constellation collisions, isolating the inter-constellation effect.
        "}.to_string();
    tab.settings.sat_radius = 3.0;
    tab.settings.show_links = false;
    tab.settings.speed = 50.0;
    tab.settings.single_color = true;
    for (alt_diff, thresh, label) in [
        (15.0, 20.0, "Crossing altitudes"),
        (500.0, 20.0, "Separated altitudes"),
    ] {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(format!("Earth ({})", label));
        planet.celestial_body = CelestialBody::Earth;
        planet.kessler.enabled = true;
        planet.kessler.collision_threshold_km = thresh;
        planet.kessler.fragments_per_collision = 15;
        planet.kessler.max_debris = 5000;
        let mut cons = ConstellationConfig::new(1);
        cons.sats_per_plane = 20;
        cons.num_planes = 10;
        cons.inclination = 53.0;
        cons.altitude_km = 550.0;
        cons.walker_type = WalkerType::Delta;
        cons.phasing = 0.5;
        planet.constellations.push(cons);
        let mut cons2 = ConstellationConfig::new(7);
        cons2.sats_per_plane = 15;
        cons2.num_planes = 8;
        cons2.inclination = 97.6;
        cons2.altitude_km = 550.0 + alt_diff;
        cons2.walker_type = WalkerType::Star;
        cons2.phasing = 0.5;
        // Offset the second constellation in RAAN and argument of periapsis so
        // its satellites don't start at the same longitudes/phases as the
        // first — prevents t=0 pairs from spawning right on top of each other.
        cons2.raan_offset = 12.0;
        cons2.arg_periapsis = 17.0;
        planet.constellations.push(cons2);
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn propagator_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Propagator: Keplerian vs J2".to_string());
    tab.title = "Propagator: Keplerian vs J2".to_string();
    tab.description = indoc::indoc! {"
            Two identical constellations propagated with different force models (Keplerian vs J2).

            A **propagator** is the model used to advance a satellite's state through time. Real orbits are perturbed by non-spherical gravity, drag, and third-body effects; different propagators capture different subsets of these.

            • **Keplerian** assumes the Earth is a point mass, giving fixed, closed elliptical orbits. Simple and fast, but unrealistic over long timescales.
            • **J2** adds the largest perturbation; Earth's equatorial bulge (J2 coefficient ≈ 0.00108). This causes the orbital plane to precess (RAAN drift) and the argument of periapsis to rotate.

            At 5000× speed you can watch the J2 planes visibly drift apart from the Keplerian ones within seconds.
        "}.to_string();
    tab.settings.sat_radius = 1.0;
    tab.settings.link_width = 2.0;
    tab.settings.show_links = false;
    tab.settings.show_orbits = true;
    tab.settings.earth_fixed_camera = true;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(60.0_f64.to_radians(), 0.0);
    tab.settings.speed = 5000.0;
    tab.settings.single_color = true;
    tab.settings.reset_time_on_switch = true;
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    for (prop, label, color) in [
        (crate::config::Propagator::Keplerian, "Keplerian", 1),
        (crate::config::Propagator::J2, "J2", 7),
    ] {
        let mut cons = ConstellationConfig::new(color);
        cons.sats_per_plane = 1;
        cons.num_planes = 6;
        cons.inclination = 60.0;
        cons.altitude_km = 500.0;
        cons.walker_type = WalkerType::Delta;
        cons.propagator = prop;
        cons.label = Some(label.to_string());
        planet.constellations.push(cons);
    }
    tab.planets.push(planet);
    v.tabs.push(tab);
}

fn radiation_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Radiation & Geomagnetic Field".to_string());
    tab.title = "Radiation & Geomagnetic Field".to_string();
    tab.description = indoc::indoc! {"
            Two Earths showing the geomagnetic field strength (left) and the trapped-particle radiation belts (right).

            Earth's magnetic field traps high-energy charged particles from the solar wind into two toroidal regions around the planet, known as the **Van Allen radiation belts**.

            • The **geomagnetic field** (left) is modeled by the IGRF, which represents Earth's field as a dipole offset from the planet's center with higher-order corrections. **Blue** indicates a weaker field, **yellow** indicates a stronger field.
            • The **radiation belts** (right) are two concentric shells: **yellow** marks the inner **proton belt** (~1000-6000 km), **blue** marks the outer **electron belt** (~13000-60000 km).

            Note that the **magnetic poles** are tilted with respect to the **geographic poles**, and the magnetic dipole is offset slightly from Earth's center. This creates the **South Atlantic Anomaly (SAA)**: a region over South America and the South Atlantic where the magnetic field is weakest, allowing trapped radiation to dip much closer to the surface. Satellites passing through the SAA receive the strongest radiation dose of any point in their orbit, and many spacecraft mute sensitive instruments while crossing it.

            LEO satellites typically sit below both belts; MEO (GPS) sits between them; GEO sits above.
        "}.to_string();
    tab.settings.sat_radius = 0.0;
    tab.settings.show_links = false;
    tab.settings.show_radiation_belts = true;
    tab.settings.show_magnetic_axis = true;
    tab.settings.show_axes = true;
    tab.settings.earth_fixed_camera = false;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(0.0, (-90.0_f64).to_radians());
    tab.settings.zoom = 10000.0 / 12000.0;
    tab.settings.auto_rotate = true;
    tab.settings.auto_rotate_speed = 3.0;
    tab.settings.auto_rotate_axis_lat = 0.0;
    tab.settings.auto_rotate_axis_lon = 0.0;
    {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(
            "Geomagnetic Field at 500 km (brighter = stronger magnetic field)".to_string(),
        );
        planet.celestial_body = CelestialBody::Earth;
        planet.radiation.show_heatmap_sphere = true;
        planet.radiation.heatmap_mode = crate::config::HeatmapMode::IgrfField;
        planet.radiation.heatmap_altitude_km = 500.0;
        tab.planets.push(planet);
    }
    {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(
            "Radiation Belts at 500 km (brighter = higher radiation)".to_string(),
        );
        planet.celestial_body = CelestialBody::Earth;
        planet.radiation.show_heatmap_sphere = true;
        planet.radiation.heatmap_mode = crate::config::HeatmapMode::IgrfRadiation;
        planet.radiation.heatmap_altitude_km = 500.0;
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn starlink_iris_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Starlink vs IRIS²".to_string());
    tab.title = "Starlink vs IRIS²".to_string();
    tab.description = indoc::indoc! {"
            Two real-world constellations side by side: SpaceX's Starlink (LEO only) and the European IRIS² (LEO + MEO).

            Modern broadband mega-constellations trade off coverage, latency, and satellite count with different architectures.

            • **Starlink** uses four LEO shells (~540–570 km, ~4400 satellites) at inclinations 53°, 53.2°, 70°, and 97.6°. LEO gives low latency but each satellite covers a small footprint, so many are needed.
            • **IRIS²** combines a LEO shell (~1200 km, 264 sats at 87°) with a MEO shell (~8000 km, 18 sats at 56°). MEO's larger footprint fills coverage gaps with far fewer satellites, at the cost of higher latency.
        "}.to_string();
    tab.settings.sat_radius = 4.0;
    tab.settings.link_width = 1.0;
    tab.settings.show_links = false;
    tab.settings.zoom = 10000.0 / 15000.0;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(0.0, 0.0);
    tab.settings.auto_rotate = true;
    tab.settings.auto_rotate_speed = 3.0;
    tab.settings.auto_rotate_axis_lat = 0.0;
    tab.settings.auto_rotate_axis_lon = 0.0;
    {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new("Starlink".to_string());
        planet.celestial_body = CelestialBody::Earth;
        for (spp, np, alt, inc, wt, color, name) in [
            (22, 72, 550.0, 53.0, WalkerType::Delta, 0, "Shell 1 (53°)"),
            (20, 36, 540.0, 53.2, WalkerType::Delta, 1, "Shell 2 (53.2°)"),
            (58, 6, 570.0, 70.0, WalkerType::Delta, 2, "Shell 3 (70°)"),
            (43, 4, 560.0, 97.6, WalkerType::Star, 3, "Shell 4 (SSO)"),
        ] {
            let mut cons = ConstellationConfig::new(color);
            cons.sats_per_plane = spp;
            cons.num_planes = np;
            cons.altitude_km = alt;
            cons.inclination = inc;
            cons.walker_type = wt;
            cons.label = Some(name.to_string());
            planet.constellations.push(cons);
        }
        tab.planets.push(planet);
    }
    {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new("IRIS²".to_string());
        planet.celestial_body = CelestialBody::Earth;
        for (spp, np, alt, inc, wt, color, name) in [
            (22, 12, 1200.0, 87.0, WalkerType::Star, 0, "LEO (87°)"),
            (6, 2, 8062.0, 56.0, WalkerType::Delta, 1, "MEO (56°)"),
        ] {
            let mut cons = ConstellationConfig::new(color);
            cons.sats_per_plane = spp;
            cons.num_planes = np;
            cons.altitude_km = alt;
            cons.inclination = inc;
            cons.walker_type = wt;
            cons.label = Some(name.to_string());
            planet.constellations.push(cons);
        }
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn oblateness_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Oblateness: Earth, Jupiter, Saturn".to_string());
    tab.title = "Oblateness: Earth, Jupiter, Saturn".to_string();
    tab.description = indoc::indoc! {"
            Three planets with progressively larger equatorial bulges (Earth, Jupiter, Saturn).

            **Oblateness** (or flattening) is the fraction by which a planet's polar radius is smaller than its equatorial radius. A rotating fluid body bulges at the equator because of centrifugal force; the faster it spins and the lower its gravity, the larger the bulge.

            The bulge pulls on orbiting satellites unevenly; instead of a pure central gravity, the orbit feels a small extra torque from the extra mass at the equator. Over many orbits this causes the orbital plane to slowly rotate around the planet's spin axis, an effect called **nodal precession**.

            • **Earth** is only 0.3% oblate; nearly spherical, yet the effect is still large enough to dominate LEO orbital perturbations.
            • **Jupiter** is 6.5% oblate; its rapid 10-hour day creates a visibly squashed profile.
            • **Saturn** is 9.8% oblate, the most oblate of the planets, thanks to a 10.7-hour day and relatively low density.
        "}.to_string();
    tab.settings.sat_radius = 3.0;
    tab.settings.show_links = false;
    tab.settings.show_polar_circle = true;
    tab.settings.auto_rotate = true;
    tab.settings.auto_rotate_speed = 3.0;
    for (body, label) in [
        (CelestialBody::Earth, "Earth (0.3% oblate)"),
        (CelestialBody::Jupiter, "Jupiter (6.5% oblate)"),
        (CelestialBody::Saturn, "Saturn (9.8% oblate)"),
    ] {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(label.to_string());
        planet.celestial_body = body;
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn solar_system_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Solar System".to_string());
    tab.title = "Solar System".to_string();
    tab.description = indoc::indoc! {"
            The eight planets orbiting the Sun, with a circular calendar overlay marking the current date.

            The **Solar System** is the gravitationally bound system of the Sun, its planets, and the other bodies that orbit it.

            Planets move in elliptical orbits obeying Kepler's laws; closer planets orbit faster (Mercury's year is 88 days; Neptune's is 165 Earth years). Orbit sizes are shown on a compressed logarithmic scale so all eight planets remain visible at once.
        "}.to_string();
    tab.settings.view_mode = crate::config::ViewMode::SolarSystem;
    tab.settings.show_circular_calendar = true;
    tab.settings.solar_system_log_power = 1.0;
    tab.settings.solar_system_hide_bodies = true;
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    tab.planets.push(planet);
    v.tabs.push(tab);
    v.ss_auto_zoom = true;
    v.ss_auto_zoom_time = 0.0;
}

#[cfg(not(target_arch = "wasm32"))]
fn planet_sizes_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Planet Sizes".to_string());
    tab.title = "Planet Sizes".to_string();
    tab.description = indoc::indoc! {"
            The eight planets of the solar system lined up at true relative size, with the Sun alongside for scale.

            Planetary radii span four orders of magnitude; from **Mercury** at 2440 km to the **Sun** at 696000 km; a range hard to grasp without a direct visual comparison.

            The **gas giants** (Jupiter, Saturn, Uranus, Neptune) dwarf the **rocky planets** (Mercury, Venus, Earth, Mars), and the Sun in turn dwarfs them all: you could fit over a million Earths inside it.
        "}.to_string();
    tab.settings.view_mode = crate::config::ViewMode::PlanetSizes;
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    tab.planets.push(planet);
    v.tabs.push(tab);
    v.planet_sizes_auto_zoom = true;
    v.planet_sizes_auto_time = 0.0;
}

fn starlink_tle_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Starlink: Simulated vs Real".to_string());
    tab.title = "Starlink: Simulated vs Real".to_string();
    tab.description = indoc::indoc! {"
            An idealised Walker Delta simulation of Starlink (left) compared to the real constellation propagated from live TLE data (right).

            The idealised model uses clean Walker shells with uniform spacing. The real constellation shows launch-and-deployment artefacts: satellites bunched at inject altitude, raising through orbit-raising phases, and gaps where failures have occurred.
        "}.to_string();
    tab.settings.show_links = false;
    tab.planet_counter += 1;
    let mut planet_sim = PlanetConfig::new("Simulated".to_string());
    planet_sim.celestial_body = CelestialBody::Earth;
    for (spp, np, alt, inc, wt, color, name) in [
        (22, 72, 550.0, 53.0, WalkerType::Delta, 0, "Shell 1 (53°)"),
        (20, 36, 540.0, 53.2, WalkerType::Delta, 1, "Shell 2 (53.2°)"),
        (58, 6, 570.0, 70.0, WalkerType::Delta, 2, "Shell 3 (70°)"),
        (43, 4, 560.0, 97.6, WalkerType::Star, 3, "Shell 4 (SSO)"),
    ] {
        let mut cons = ConstellationConfig::new(color);
        cons.preset = Preset::Starlink;
        cons.sats_per_plane = spp;
        cons.num_planes = np;
        cons.altitude_km = alt;
        cons.inclination = inc;
        cons.walker_type = wt;
        cons.label = Some(name.to_string());
        planet_sim.constellations.push(cons);
    }
    tab.planets.push(planet_sim);
    tab.planet_counter += 1;
    let mut planet_real = PlanetConfig::new("Real TLE".to_string());
    planet_real.celestial_body = CelestialBody::Earth;
    planet_real.show_tle_window = true;
    planet_real.auto_cluster_tle = true;
    planet_real
        .tle_selections
        .insert(TlePreset::Starlink, (true, TleLoadState::NotLoaded, None));
    tab.planets.push(planet_real);
    v.tabs.push(tab);
}

fn all_tle_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Live data of all constellations".to_string());
    tab.title = "Live data of all constellations".to_string();
    tab.description = indoc::indoc! {"
            Every active operational satellite group from CelesTrak, propagated from live **TLE** data.

            **CelesTrak** is a public catalogue of satellite orbital elements maintained since 1985. Each colour here represents a different operational group; Starlink, GPS, Iridium, weather satellites, scientific spacecraft, and dozens more.

            The auto-zoom cycles between LEO (where most satellites live) and GEO (where broadcast and weather satellites hang stationary over the equator), revealing how densely populated Earth orbit has become.
        "}.to_string();
    tab.settings.sat_radius = 3.0;
    tab.settings.zoom = 10000.0 / 10000.0;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(20.0_f64.to_radians(), 0.0);
    tab.settings.auto_rotate = true;
    tab.settings.auto_rotate_speed = 3.0;
    tab.settings.auto_rotate_axis_lat = 0.0;
    tab.settings.auto_rotate_axis_lon = 0.0;
    tab.settings.auto_zoom = true;
    tab.settings.auto_zoom_min_alt = 10000.0;
    tab.settings.auto_zoom_max_alt = 35000.0;
    tab.settings.auto_zoom_duration = 20.0;
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    planet.show_tle_window = true;
    for preset in TlePreset::ALL {
        let selected = !matches!(
            preset,
            TlePreset::Last30Days
                | TlePreset::Brightest100
                | TlePreset::ActiveSats
                | TlePreset::CountrySweden
                | TlePreset::CountryEurope
                | TlePreset::CountryUsa
                | TlePreset::CountryChina
                | TlePreset::CountryIndia
        );
        planet
            .tle_selections
            .insert(preset, (selected, TleLoadState::NotLoaded, None));
    }
    tab.planets.push(planet);
    v.tabs.push(tab);
}

fn projections_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Map Projections".to_string());
    tab.title = "Map Projections".to_string();
    tab.description = indoc::indoc! {"
            Real Starlink satellites rendered in six different **map projections** side by side.

            Every map projection distorts the Earth's spherical surface in some way to fit it on a flat (or near-flat) canvas, and different projections preserve different properties.

            • **Orthographic** shows a hemisphere as seen from infinity. Shapes near the limb are compressed but the view is intuitive.
            • **Mercator** preserves angles and shapes locally (conformal), so compass directions are straight lines. Area explodes near the poles (Greenland looks huge).
            • **Mollweide** is equal-area; country areas are proportional to reality, at the cost of distorting shapes.
            • **Azimuthal Equidistant** preserves distance from the center; common for polar and radio-range maps, famously used on the UN flag.
            • **HEALPix** tiles the sphere into equal-area pixels along four rhombic slices; common in cosmology for all-sky survey data.
            • **Sinusoidal** is equal-area with straight parallels; the outer profile is a pair of sine curves.
        "}.to_string();
    tab.settings.sat_radius = 0.5;
    tab.settings.earth_fixed_camera = true;
    tab.settings.zoom = 1.0;
    let projections = [
        (
            crate::projection::ProjectionKind::Orthographic,
            "Orthographic",
        ),
        (crate::projection::ProjectionKind::Mercator, "Mercator"),
        (crate::projection::ProjectionKind::Mollweide, "Mollweide"),
        (
            crate::projection::ProjectionKind::AzimuthalEquidistant,
            "Azimuthal Equidistant",
        ),
        (crate::projection::ProjectionKind::HEALPix, "HEALPix"),
        (crate::projection::ProjectionKind::Sinusoidal, "Sinusoidal"),
    ];
    for (proj, label) in projections {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(label.to_string());
        planet.celestial_body = CelestialBody::Earth;
        planet.projection_override = Some(proj);
        planet.show_tle_window = false;
        planet.auto_cluster_tle = true;
        planet
            .tle_selections
            .insert(TlePreset::Starlink, (true, TleLoadState::NotLoaded, None));
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn all_tle_map_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Live data of all constellations (map)".to_string());
    tab.title = "Live data of all constellations (map)".to_string();
    tab.description = indoc::indoc! {"
            The same live TLE catalogue, projected onto an equirectangular world map.

            An **equirectangular projection** lays longitude and latitude on perpendicular axes, so each satellite's instantaneous sub-satellite point is plotted directly. Distances are distorted (badly at the poles), but the layout makes it easy to read which constellations sit at which altitudes and inclinations.

            The sinusoidal bands traced by each orbital plane are the familiar **ground-track waves**. Altitude shells show up as distinct horizontal bands; polar-orbit constellations fill wide latitude ranges, while low-inclination shells stay near the equator.
        "}.to_string();
    tab.settings.sat_radius = 1.5;
    tab.settings.planet_projection = crate::projection::ProjectionKind::Equirectangular;
    tab.settings.earth_fixed_camera = true;
    tab.settings.zoom = 1.0;
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    planet.show_tle_window = true;
    for preset in TlePreset::ALL {
        let selected = !matches!(
            preset,
            TlePreset::Last30Days
                | TlePreset::Brightest100
                | TlePreset::ActiveSats
                | TlePreset::CountrySweden
                | TlePreset::CountryEurope
                | TlePreset::CountryUsa
                | TlePreset::CountryChina
                | TlePreset::CountryIndia
        );
        planet
            .tle_selections
            .insert(preset, (selected, TleLoadState::NotLoaded, None));
    }
    tab.planets.push(planet);
    v.tabs.push(tab);
}

#[cfg(not(target_arch = "wasm32"))]
fn countries_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Live data by country/region".to_string());
    tab.title = "Live data by country/region".to_string();
    tab.description = indoc::indoc! {"
            Active satellites broken out by country/region of ownership.

            The list is built from CelesTrak's **SATCAT** catalogue, which records the registered owner of every tracked object, intersected with the live active-satellite TLE feed. Coverage is approximate — joint missions and commercial operators registered abroad don't always land where you'd expect.

            • **Sweden** — Odin, MATS and a handful of national missions.
            • **Europe** — ESA, EUMETSAT, EUTELSAT and the major member states (FR, DE, IT, ES, NL, etc.).
            • **USA** — by far the largest fleet (Starlink alone dwarfs every other operator).
            • **China** — Beidou, Yaogan, Tiangong, Gaofen and a fast-growing commercial sector.
            • **India** — ISRO's NavIC, Cartosat, RISAT, Oceansat constellations.
        "}.to_string();
    tab.settings.sat_radius = 3.0;
    tab.settings.zoom = 10000.0 / 10000.0;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(20.0_f64.to_radians(), 0.0);
    tab.settings.auto_rotate = true;
    tab.settings.auto_rotate_speed = 3.0;
    tab.settings.auto_rotate_axis_lat = 0.0;
    tab.settings.auto_rotate_axis_lon = 0.0;
    tab.settings.show_links = false;

    let countries = [
        (TlePreset::CountrySweden, "Sweden"),
        (TlePreset::CountryEurope, "Europe"),
        (TlePreset::CountryUsa, "USA"),
        (TlePreset::CountryChina, "China"),
        (TlePreset::CountryIndia, "India"),
    ];
    for (preset, label) in countries {
        tab.planet_counter += 1;
        let mut planet = PlanetConfig::new(label.to_string());
        planet.celestial_body = CelestialBody::Earth;
        planet.show_tle_window = false;
        planet
            .tle_selections
            .insert(preset, (true, TleLoadState::NotLoaded, None));
        tab.planets.push(planet);
    }
    v.tabs.push(tab);
}

fn debris_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Live data of debris".to_string());
    tab.title = "Live data of debris".to_string();
    tab.description = indoc::indoc! {"
            Tracked debris fragments in Earth orbit, propagated from real TLE data.

            A **TLE** (Two-Line Element set) is a compact text encoding of a satellite's orbital state at a given epoch, published daily by organisations like **CelesTrak** and the U.S. Space Force. Feeding a TLE through the **SGP4** propagator reproduces the satellite's current position in real time.

            **Orbital debris** includes everything from defunct satellites and spent rocket stages to fragments from past collisions and anti-satellite weapon tests.

            • **Fengyun 1C** debris (2007); a Chinese ASAT test generated ~3000 trackable fragments.
            • **Iridium 33 / Cosmos 2251** debris (2009); the first accidental collision between two intact satellites produced ~2000 fragments.
            • **Cosmos 1408** debris (2021); another ASAT test created ~1500 trackable fragments.

            Only tracked fragments (typically ≥10 cm) appear here. Estimates of untracked small debris run into the hundreds of millions.
        "}.to_string();
    tab.settings.sat_radius = 2.0;
    tab.settings.zoom = 10000.0 / 10000.0;
    tab.settings.rotation = crate::math::lat_lon_to_matrix(20.0_f64.to_radians(), 0.0);
    tab.settings.auto_rotate = true;
    tab.settings.auto_rotate_speed = 3.0;
    tab.settings.auto_rotate_axis_lat = 0.0;
    tab.settings.auto_rotate_axis_lon = 0.0;
    tab.settings.zoom = 10000.0 / 11000.0;
    tab.settings.tle_monochrome = true;
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    planet.show_tle_window = true;
    for preset in TlePreset::ALL {
        let selected = preset.is_debris();
        planet
            .tle_selections
            .insert(preset, (selected, TleLoadState::NotLoaded, None));
    }
    tab.planets.push(planet);
    v.tabs.push(tab);
}

fn iss_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty("Live ISS Tracking".to_string());
    tab.title = "Live ISS Tracking".to_string();
    tab.description = indoc::indoc! {"
            The International Space Station tracked from live TLE data, with a nadir-pointing camera window following its ground track.

            The **International Space Station (ISS)** is a continuously inhabited orbital laboratory operated jointly by NASA, Roscosmos, ESA, JAXA, and CSA since 2000.

            It orbits at ~420 km altitude with a 51.6° inclination; chosen to be reachable from both Cape Canaveral and Baikonur; completing one orbit every ~93 minutes.
        "}.to_string();
    tab.settings.sat_radius = 2.0;
    tab.settings.zoom = 10000.0 / 5000.0;
    tab.settings.camera_mode = crate::config::CameraMode::TrackSatellite;
    tab.settings.show_camera_windows = true;
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    planet.show_tle_window = true;
    planet
        .tle_selections
        .insert(TlePreset::Stations, (true, TleLoadState::NotLoaded, None));
    v.camera_id_counter += 1;
    planet
        .satellite_cameras
        .push(crate::config::SatelliteCamera {
            id: v.camera_id_counter,
            label: "ISS".to_string(),
            constellation_idx: usize::MAX,
            plane: 0,
            sat_index: 0,
            screen_pos: None,
        });
    tab.planets.push(planet);
    v.tabs.push(tab);
}

pub(crate) fn spacecomp_demo(v: &mut ViewerState) {
    v.tab_counter += 1;
    v.tabs.push(spacecomp_demo_tab(
        "SpaceCoMP".to_string(),
        "SpaceCoMP".to_string(),
        0.0,
    ));
}

fn spacecomp_demo_tab(name: String, title: String, time: f64) -> TabConfig {
    let mut tab = TabConfig::new_empty(name);
    tab.title = title;
    tab.description = indoc::indoc! {"
            A distributed-computing scenario: image an area of interest, process the data on-orbit, then deliver the result to a ground station.

            **SpaceComp** assigns specialised roles to satellites participating in a job:

            • **Collectors** are satellites currently over the **area of interest (AOI)** that capture the raw data.
            • **Mappers** receive data from collectors and do the heavy per-tile processing.
            • A **Reducer** aggregates mapper outputs into the final product.
            • A **Line-of-Sight** satellite over the **ground station (GS)** acts as the downlink relay.

            Roles are reassigned every cycle as the constellation rotates; the satellites themselves don't know in advance which job they'll play.
        "}.to_string();
    tab.settings.time = time;
    tab.settings.speed = 10.0;
    tab.settings.earth_fixed_camera = true;
    tab.settings.rotation =
        crate::math::lat_lon_to_matrix(8.5_f64.to_radians(), 15.0_f64.to_radians());
    tab.settings.zoom = 10000.0 / 2200.0;
    tab.settings.sat_radius = 2.0;
    tab.settings.show_asc_desc_colors = true;
    tab.planet_counter += 1;
    let mut planet = PlanetConfig::new("Earth".to_string());
    planet.celestial_body = CelestialBody::Earth;
    let mut cons = ConstellationConfig::new(0);
    cons.sats_per_plane = 71;
    cons.num_planes = 49;
    cons.inclination = 53.0;
    cons.walker_type = WalkerType::Star;
    cons.phasing = 1.0;
    cons.raan_offset = 180.0;
    planet.constellations.push(cons);
    planet.ground_stations.push(GroundStation {
        name: "Ground Station".to_string(),
        lat: 18.840,
        lon: 5.476,
        radius_km: 300.0,
        color: egui::Color32::from_rgb(255, 100, 100),
        selected: false,
    });
    planet.areas_of_interest.push(AreaOfInterest {
        name: "Area of Interest".to_string(),
        lat: 0.339,
        lon: 19.319,
        radius_km: 500.0,
        color: egui::Color32::from_rgba_unmultiplied(130, 230, 130, 110),
        ground_station_idx: Some(0),
        job_mode: AoiJobMode::SpaceComp,
        job_n: 3,
        reducer_placement: crate::config::SpaceCompReducerPlacement::NearMappers,
        selected: false,
    });
    tab.planets.push(planet);
    tab
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Presentation {
    SpaceCoMP,
}

impl Presentation {
    pub(crate) const ALL: &'static [Presentation] = &[Presentation::SpaceCoMP];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Presentation::SpaceCoMP => "SpaceCoMP",
        }
    }

    fn build(self, v: &mut ViewerState) {
        match self {
            Presentation::SpaceCoMP => {
                spacecomp_slide_tabs(v, crate::slides::SPACECOMP_PRIMER, 1..35);
                spacecomp_simulation_tabs(v, 35..41);
                spacecomp_slide_tabs(
                    v,
                    crate::slides::SPACECOMP_PRIMER,
                    41..crate::slides::total_slide_count() + 1,
                );
            }
        }
    }
}

fn spacecomp_simulation_tabs(v: &mut ViewerState, slide_numbers: std::ops::Range<usize>) {
    for slide_number in slide_numbers {
        spacecomp_presentation_demo_tab(v, slide_number);
    }
}

fn spacecomp_presentation_demo_tab(v: &mut ViewerState, slide_number: usize) {
    match slide_number {
        35 => inclination_demo(v),
        36 => walker_demo(v),
        37 => isl_demo(v),
        38 => torus_demo(v),
        39 => routing_demo(v),
        40 => {
            v.tab_counter += 1;
            v.tabs.push(spacecomp_demo_tab(
                format!("Slide {}", slide_number),
                String::new(),
                0.0,
            ));
        }
        _ => return,
    }

    if let Some(tab) = v.tabs.last_mut() {
        tab.name = format!("Slide {}", slide_number);
        tab.title = String::new();
        tab.description = String::new();
        tab.presentation_slide_number = Some(slide_number);
    }
}

fn spacecomp_slide_tabs(
    v: &mut ViewerState,
    deck_id: crate::slides::DeckId,
    slide_numbers: std::ops::Range<usize>,
) {
    for slide_number in slide_numbers {
        let slide_idx = slide_number.saturating_sub(1);
        spacecomp_slides_tab(
            v,
            crate::slides::SlideDeck::range(deck_id, slide_idx..slide_idx + 1),
            &format!("Slide {}", slide_number),
        );
    }
}

fn spacecomp_slides_tab(v: &mut ViewerState, deck: crate::slides::SlideDeck, name: &str) {
    v.tab_counter += 1;
    let mut tab = TabConfig::new_empty(name.to_string());
    tab.title = String::new();
    tab.slides = Some(deck);
    v.tabs.push(tab);
}

impl App {
    pub(crate) fn setup_demo(&mut self) {
        self.setup_tabs(|v| {
            // Orbit fundamentals
            inclination_demo(v);
            altitude_demo(v);
            eccentricity_demo(v);
            ground_track_demo(v);
            // Planet shape & propagation
            oblateness_demo(v);
            propagator_demo(v);
            // Constellation design
            walker_demo(v);
            phasing_demo(v);
            kessler_demo(v);
            debris_demo(v);
            coverage_demo(v);
            // Networking
            isl_demo(v);
            torus_demo(v);
            routing_demo(v);
            spacecomp_demo(v);
            // Hazards
            radiation_demo(v);
            // Real constellations
            sso_demo(v);
            starlink_iris_demo(v);
            starlink_tle_demo(v);
            all_tle_demo(v);
            all_tle_map_demo(v);
            #[cfg(not(target_arch = "wasm32"))]
            countries_demo(v);
            projections_demo(v);
            iss_demo(v);
            // Context & scale
            solar_system_demo(v);
            #[cfg(not(target_arch = "wasm32"))]
            planet_sizes_demo(v);
        });
    }

    pub(crate) fn setup_presentation(
        &mut self,
        presentation: Presentation,
        ctx: &eframe::egui::Context,
    ) {
        self.setup_tabs(|v| presentation.build(v));
        match presentation {
            Presentation::SpaceCoMP => {
                crate::slides::warm_browser_cache(crate::slides::SPACECOMP_PRIMER)
            }
        }
        self.viewer.slide_textures.clear();
        ctx.request_repaint();
    }

    fn setup_tabs<F: FnOnce(&mut ViewerState)>(&mut self, build: F) {
        let v = &mut self.viewer;

        let mut tle_cache: std::collections::HashMap<TlePreset, Vec<crate::tle::TleSatellite>> =
            std::collections::HashMap::new();
        for tab in &v.tabs {
            for planet in &tab.planets {
                for (preset, (_, state, _)) in &planet.tle_selections {
                    if let TleLoadState::Loaded { satellites } = state {
                        tle_cache
                            .entry(*preset)
                            .or_insert_with(|| satellites.clone());
                    }
                }
            }
        }

        v.tabs.clear();
        v.tab_counter = 0;

        build(v);

        for tab in &mut v.tabs {
            if tab.settings.auto_rotate {
                tab.settings.initial_rotation = Some(tab.settings.rotation);
            }
        }

        self.dock_state = DockState::new(vec![0]);
        for i in 1..v.tabs.len() {
            self.dock_state.push_to_focused_leaf(i);
        }
        self.dock_state.set_active_tab((
            egui_dock::SurfaceIndex::main(),
            egui_dock::NodeIndex::root(),
            egui_dock::TabIndex(0),
        ));

        // Restore cached TLE data and fetch missing presets
        let mut fetched = std::collections::HashSet::new();
        for tab in &mut v.tabs {
            for planet in &mut tab.planets {
                for (preset, (selected, state, _shells)) in &mut planet.tle_selections {
                    if !*selected {
                        continue;
                    }
                    if let Some(cached_sats) = tle_cache.get(preset) {
                        *state = TleLoadState::Loaded {
                            satellites: cached_sats.clone(),
                        };
                        continue;
                    }
                    if !matches!(state, TleLoadState::NotLoaded) {
                        continue;
                    }
                    *state = TleLoadState::Loading;
                    if !fetched.insert(*preset) {
                        continue;
                    }
                    let preset_copy = *preset;

                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let tx = v.tle_fetch_tx.clone();
                        if let Some(owners) = preset.country_owners() {
                            std::thread::spawn(move || {
                                let result = crate::tle::fetch_tle_by_country(owners);
                                let _ = tx.send((preset_copy, result));
                            });
                        } else {
                            let url = preset.url().to_string();
                            std::thread::spawn(move || {
                                let result = crate::tle::fetch_tle_data(&url);
                                let _ = tx.send((preset_copy, result));
                            });
                        }
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        use crate::tle::TLE_FETCH_RESULT;
                        if let Some(owners) = preset.country_owners() {
                            wasm_bindgen_futures::spawn_local(async move {
                                let result = crate::tle::fetch_tle_by_country_async(owners).await;
                                TLE_FETCH_RESULT.with(|cell| {
                                    cell.borrow_mut().push((preset_copy, result));
                                });
                            });
                        } else {
                            let url = preset.url().to_string();
                            wasm_bindgen_futures::spawn_local(async move {
                                let result = match crate::tle::fetch_tle_text(&url).await {
                                    Ok(text) => crate::tle::parse_tle_data_async(&text).await,
                                    Err(e) => Err(e),
                                };
                                TLE_FETCH_RESULT.with(|cell| {
                                    cell.borrow_mut().push((preset_copy, result));
                                });
                            });
                        }
                    }
                }
            }
        }

        v.auto_cycle_tabs = false;
        v.cycle_interval = 30.0;
        v.last_cycle_time = 0.0;
        v.show_tab_info = true;

        for tab in &mut v.tabs {
            let speed = tab.settings.speed;
            let suffix = format!(" ({}x speed)", format_int(speed));
            if !tab.title.is_empty() {
                tab.title.push_str(&suffix);
            }
        }
    }
}

fn format_int(speed: f64) -> String {
    if speed.fract().abs() < 1e-6 {
        format!("{}", speed as i64)
    } else {
        format!("{:.1}", speed)
    }
}
