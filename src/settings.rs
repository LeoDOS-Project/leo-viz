//! Side-panel settings UI for camera, animation, and display options.

use crate::celestial::{CelestialBody, TextureResolution};
use crate::math::{matrix_to_lat_lon, lat_lon_to_matrix};
use crate::time::body_rotation_angle;
use crate::ViewerState;
use eframe::egui;
use nalgebra::Matrix3;
use chrono::{Duration, Local, Utc};

impl ViewerState {
    pub(crate) fn show_settings(&mut self, ui: &mut egui::Ui) {
        let current_body = self.tabs.first()
            .and_then(|t| t.planets.first())
            .map(|p| p.celestial_body)
            .unwrap_or(CelestialBody::Earth);

        let active = self.active_tab_idx;
        let current_gmst = self.current_gmst;

        {
            let s = &mut self.tabs[active].settings;
            let (time_ref, speed_ref, animate_ref) =
                (&mut s.time, &mut s.speed, &mut s.animate);

            ui.label(egui::RichText::new("Simulation").strong());
            ui.horizontal(|ui| {
                ui.label("Speed:");
                let abs_speed = speed_ref.abs();
                let drag_speed = if abs_speed > 31_536_000.0 { 100_000.0 } else if abs_speed > 86400.0 { 1000.0 } else if abs_speed > 3600.0 { 100.0 } else { 1.0 };
                ui.add(egui::DragValue::new(speed_ref).range(-3_153_600_000.0..=3_153_600_000.0).speed(drag_speed))
                    .on_hover_text("Simulation speed multiplier");
                if ui.button("⏪").on_hover_text("Reverse direction").clicked() {
                    *speed_ref = -*speed_ref;
                }
                let pause_label = if *animate_ref { "⏸" } else { "▶" };
                if ui.button(pause_label).on_hover_text("Pause/resume simulation").clicked() {
                    *animate_ref = !*animate_ref;
                }
                if abs_speed > 1.0 {
                    let label = if abs_speed >= 31_536_000.0 {
                        format!("{:.1} earth years/s", abs_speed / 31_536_000.0)
                    } else if abs_speed >= 2_592_000.0 {
                        format!("{:.1} earth months/s", abs_speed / 2_592_000.0)
                    } else if abs_speed >= 86400.0 {
                        format!("{:.1} earth days/s", abs_speed / 86400.0)
                    } else if abs_speed >= 3600.0 {
                        format!("{:.1} earth hours/s", abs_speed / 3600.0)
                    } else if abs_speed >= 60.0 {
                        format!("{:.1} earth minutes/s", abs_speed / 60.0)
                    } else {
                        format!("{:.1} earth seconds/s", abs_speed)
                    };
                    ui.label(egui::RichText::new(label).weak());
                }
            });
            let start = self.start_timestamp;
            let real_timestamp = start + Duration::seconds(self.real_time as i64);
            let current_ts = start + Duration::seconds(*time_ref as i64);
            {
                use chrono::Timelike;
                use chrono::Datelike;
                let local = current_ts.with_timezone(&Local);
                let orig_time = *time_ref;
                let mut t_sec = *time_ref;
                let mut t_min = *time_ref;
                let mut t_hour = *time_ref;
                let mut t_day = *time_ref;
                let total_months = local.year() as f64 * 12.0 + local.month() as f64 - 1.0;
                let mut t_month = total_months;
                let mut t_year = total_months;
                let fmt_component = |secs: f64, f: fn(&chrono::DateTime<Local>) -> String| -> String {
                    let ts = (start + Duration::seconds(secs as i64)).with_timezone(&Local);
                    f(&ts)
                };
                ui.horizontal(|ui| {
                    ui.label("Time:");
                    ui.add(egui::DragValue::new(&mut t_hour)
                        .speed(360.0)
                        .custom_formatter(|s, _| fmt_component(s, |t| format!("{:02}", t.hour())))
                        .custom_parser(move |input| {
                            let h = input.parse::<u32>().ok()?.min(23);
                            let delta = (h as i64 - local.hour() as i64) * 3600;
                            Some(orig_time + delta as f64)
                        }));
                    ui.label(":");
                    ui.add(egui::DragValue::new(&mut t_min)
                        .speed(6.0)
                        .custom_formatter(|s, _| fmt_component(s, |t| format!("{:02}", t.minute())))
                        .custom_parser(move |input| {
                            let m = input.parse::<u32>().ok()?.min(59);
                            let delta = (m as i64 - local.minute() as i64) * 60;
                            Some(orig_time + delta as f64)
                        }));
                    ui.label(":");
                    ui.add(egui::DragValue::new(&mut t_sec)
                        .speed(0.1)
                        .custom_formatter(|s, _| fmt_component(s, |t| format!("{:02}", t.second())))
                        .custom_parser(move |input| {
                            let s = input.parse::<u32>().ok()?.min(59);
                            let delta = s as i64 - local.second() as i64;
                            Some(orig_time + delta as f64)
                        }));
                });
                ui.horizontal(|ui| {
                    ui.label("Date:");
                    ui.add(egui::DragValue::new(&mut t_day)
                        .speed(8640.0)
                        .custom_formatter(|s, _| fmt_component(s, |t| format!("{:02}", t.day())))
                        .custom_parser(move |input| {
                            let d = input.parse::<u32>().ok()?.clamp(1, 31);
                            let delta = (d as i64 - local.day() as i64) * 86400;
                            Some(orig_time + delta as f64)
                        }));
                    ui.label("/");
                    ui.add(egui::DragValue::new(&mut t_month)
                        .speed(0.1)
                        .custom_formatter(|v, _| {
                            let m = (v as i32).rem_euclid(12) + 1;
                            format!("{:02}", m)
                        })
                        .custom_parser(move |input| {
                            let m: i32 = input.parse().ok()?;
                            Some(local.year() as f64 * 12.0 + m.clamp(1, 12) as f64 - 1.0)
                        }));
                    ui.label("/");
                    ui.add(egui::DragValue::new(&mut t_year)
                        .speed(1.2)
                        .custom_formatter(|v, _| {
                            let y = (v / 12.0).floor() as i32;
                            format!("{}", y)
                        })
                        .custom_parser(move |input| {
                            let y: i32 = input.parse().ok()?;
                            Some(y as f64 * 12.0 + local.month() as f64 - 1.0)
                        }));
                });
                if t_sec != *time_ref { *time_ref = t_sec; }
                else if t_min != *time_ref {
                    let d = t_min - *time_ref;
                    *time_ref += (d / 60.0).round() * 60.0;
                }
                else if t_hour != *time_ref {
                    let d = t_hour - *time_ref;
                    *time_ref += (d / 3600.0).round() * 3600.0;
                }
                else if t_day != *time_ref {
                    let d = t_day - *time_ref;
                    *time_ref += (d / 86400.0).round() * 86400.0;
                }
                else {
                    let apply_month_delta = |raw: f64, unit: f64| -> Option<i32> {
                        let d = raw - total_months;
                        if d.abs() < 0.01 { return None; }
                        Some((d / unit).round() as i32)
                    };
                    let month_delta = if t_month != total_months {
                        apply_month_delta(t_month, 1.0)
                    } else if t_year != total_months {
                        apply_month_delta(t_year, 12.0).map(|d| d * 12)
                    } else {
                        None
                    };
                    if let Some(md) = month_delta {
                        let mut m = local.month() as i32 - 1 + md;
                        let y = local.year() + m.div_euclid(12);
                        m = m.rem_euclid(12) + 1;
                        let d = local.day().min(
                            chrono::NaiveDate::from_ymd_opt(y, m as u32, 1)
                                .and_then(|d| d.checked_add_months(chrono::Months::new(1)))
                                .and_then(|d| d.pred_opt())
                                .map(|d| d.day())
                                .unwrap_or(28)
                        );
                        if let Some(date) = chrono::NaiveDate::from_ymd_opt(y, m as u32, d) {
                            if let Some(dt) = date.and_time(local.time()).and_local_timezone(Local).single() {
                                let diff = dt.with_timezone(&Utc).signed_duration_since(start);
                                *time_ref = diff.num_seconds() as f64;
                            }
                        }
                    }
                }
            }
            let real_local = real_timestamp.with_timezone(&Local);
            ui.label(format!("Real: {}", real_local.format("%H:%M:%S %d/%m/%Y %Z")));
            if ui.button("Sync time").on_hover_text("Reset to current real time").clicked() {
                *time_ref = self.real_time;
            }
        }

        let tab = &mut self.tabs[active];

        ui.separator();
        ui.label(egui::RichText::new("Display").strong());

        ui.checkbox(&mut tab.settings.render_planet, "Show planet")
            .on_hover_text("Render the 3D planet surface");
        ui.indent("planet_opts", |ui| {
            let (s, planets) = (&mut tab.settings, &mut tab.planets);
            let on = s.render_planet;
            ui.add_enabled_ui(on, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Projection:");
                    use crate::projection::ProjectionKind;
                    egui::ComboBox::from_id_salt("proj_kind")
                        .selected_text(s.planet_projection.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::Orthographic, "Orthographic");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::Equirectangular, "Equirectangular");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::Mercator, "Mercator");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::Mollweide, "Mollweide");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::Sinusoidal, "Sinusoidal");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::AzimuthalEquidistant, "Azimuthal Equidistant");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::Hammer, "Hammer");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::HEALPix, "HEALPix");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::Cassini, "Cassini");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::TransverseMercator, "UTM");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::LambertAzimuthalEqualArea, "Lambert Azimuthal");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::GallPeters, "Gall-Peters");
                            ui.selectable_value(&mut s.planet_projection, ProjectionKind::PeirceQuincuncial, "Peirce Quincuncial");
                        });
                });
            });

            ui.label(egui::RichText::new("Camera").strong());
            ui.indent("camera_opts", |ui| {
                use crate::config::CameraMode;
                let body_rotation = body_rotation_angle(current_body, s.time, current_gmst);

                ui.radio_value(&mut s.camera_mode, CameraMode::Unlocked, "Unlocked")
                    .on_hover_text("Free camera rotation");
                {
                    let unlocked = s.camera_mode == CameraMode::Unlocked;
                    ui.indent("unlocked_opts", |ui| {
                    ui.add_enabled_ui(unlocked, |ui| {
                        let (lat, base_lon) = matrix_to_lat_lon(&s.rotation);
                        let geo_lon = if s.earth_fixed_camera {
                            base_lon
                        } else {
                            let mut l = base_lon - body_rotation;
                            while l > std::f64::consts::PI { l -= 2.0 * std::f64::consts::PI; }
                            while l < -std::f64::consts::PI { l += 2.0 * std::f64::consts::PI; }
                            l
                        };
                        let mut lat_deg = lat.to_degrees();
                        let mut lon_deg = geo_lon.to_degrees();
                        ui.horizontal(|ui| {
                            ui.label("Lat:");
                            let lat_changed = ui.add(egui::DragValue::new(&mut lat_deg).speed(0.5).max_decimals(1).suffix("°")).changed();
                            ui.label("Lon:");
                            let lon_changed = ui.add(egui::DragValue::new(&mut lon_deg).speed(0.5).max_decimals(1).suffix("°")).changed();
                            ui.label("Alt:").on_hover_text("Controls the visible range of the plot");
                            let mut alt_km = 10000.0 / s.zoom;
                            if ui.add(egui::DragValue::new(&mut alt_km).range(0.5..=1000000.0).speed(100.0).suffix(" km")).changed() {
                                s.zoom = (10000.0 / alt_km).clamp(0.01, 20000.0);
                            }
                            lat_deg = lat_deg.clamp(-90.0, 90.0);
                            while lon_deg > 180.0 { lon_deg -= 360.0; }
                            while lon_deg < -180.0 { lon_deg += 360.0; }
                            if lat_changed || lon_changed {
                                let target_lon = if s.earth_fixed_camera {
                                    lon_deg.to_radians()
                                } else {
                                    lon_deg.to_radians() + body_rotation
                                };
                                s.rotation = lat_lon_to_matrix(lat_deg.to_radians(), target_lon);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Cam:").on_hover_text("Camera distance for moon/sun perspective");
                            let mut dist_m = s.moon_camera_distance_km / 1_000_000.0;
                            if ui.add(egui::DragValue::new(&mut dist_m)
                                .range(0.1..=10.0).speed(0.1).max_decimals(1).suffix(" M km")).changed() {
                                s.moon_camera_distance_km = dist_m * 1_000_000.0;
                            }
                        });
                        let was_earth_fixed = s.earth_fixed_camera;
                        ui.checkbox(&mut s.earth_fixed_camera, "Fixed Lat/Lon")
                            .on_hover_text("Lock camera to geographic coordinates");
                        if s.earth_fixed_camera != was_earth_fixed {
                            let cos_a = body_rotation.cos();
                            let sin_a = body_rotation.sin();
                            let body_y_rot = Matrix3::new(
                                cos_a, 0.0, sin_a,
                                0.0, 1.0, 0.0,
                                -sin_a, 0.0, cos_a,
                            );
                            if s.earth_fixed_camera {
                                s.rotation *= body_y_rot;
                            } else {
                                s.rotation *= body_y_rot.transpose();
                            }
                        }
                        ui.checkbox(&mut s.trackpad_rotate, "Trackpad rotate")
                            .on_hover_text("Use trackpad gestures for rotation");
                        ui.checkbox(&mut s.north_up, "North up")
                            .on_hover_text("Keep north pole pointing upward");
                        ui.horizontal(|ui| {
                            for (label, tip, lat, lon) in [
                                ("N", "View from north pole", 90.0_f64, 0.0_f64),
                                ("S", "View from south pole", -90.0, 0.0),
                                ("E", "View from 90\u{b0}E", 0.0, 90.0),
                                ("W", "View from 90\u{b0}W", 0.0, -90.0),
                                ("C", "View from 0\u{b0}N 0\u{b0}E", 0.0, 0.0),
                            ] {
                                if ui.button(label).on_hover_text(tip).clicked() {
                                    s.camera_mode = CameraMode::Unlocked;
                                    let target_lon = if s.earth_fixed_camera {
                                        lon.to_radians()
                                    } else {
                                        lon.to_radians() + body_rotation
                                    };
                                    s.rotation = lat_lon_to_matrix(lat.to_radians(), target_lon);
                                }
                            }
                            ui.separator();
                            use crate::time::{DAYS_PER_YEAR, SOLAR_DECLINATION_MAX};
                            let ts = self.start_timestamp + Duration::seconds(s.time as i64);
                            use chrono::Datelike;
                            let doy = ts.ordinal() as f64;
                            let decl = SOLAR_DECLINATION_MAX * ((360.0 / DAYS_PER_YEAR) * (doy + 10.0)).to_radians().cos();
                            let sun_geo_lon = ((doy - 80.0) * 360.0 / DAYS_PER_YEAR).to_radians() - body_rotation;
                            for (label, tip, lat_deg, lon_rad) in [
                                ("Day", "View the sunlit side", decl, sun_geo_lon),
                                ("Night", "View the dark side", -decl, sun_geo_lon + std::f64::consts::PI),
                            ] {
                                if ui.button(label).on_hover_text(tip).clicked() {
                                    let target_lon = if s.earth_fixed_camera {
                                        lon_rad
                                    } else {
                                        lon_rad + body_rotation
                                    };
                                    s.rotation = lat_lon_to_matrix(lat_deg.to_radians(), target_lon);
                                }
                            }
                        });
                    });
                    });
                }
                ui.radio_value(&mut s.camera_mode, CameraMode::TrackSatellite, "Track Satellite")
                    .on_hover_text("Follow a selected satellite");
                ui.checkbox(&mut s.show_camera_windows, "Show camera windows")
                    .on_hover_text("Display satellite camera viewports");
            });

            ui.label(egui::RichText::new("Constellations").strong());
            ui.indent("constellation_opts", |ui| {
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_orbits, "Show orbits"))
                    .on_hover_text("Draw orbital paths for each satellite");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_coverage, "Show coverage"))
                    .on_hover_text("Display ground coverage cones");
                ui.indent("coverage_opts", |ui| {
                    ui.add_enabled_ui(on && s.show_coverage, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Angle:");
                            ui.add(egui::DragValue::new(&mut s.coverage_angle)
                                .range(0.5..=70.0).speed(0.1).max_decimals(1).suffix("°"))
                                .on_hover_text("Minimum elevation angle for coverage");
                        });
                    });
                });
                {
                    let mut show_behind = !s.hide_behind_earth;
                    if ui.add_enabled(on, egui::Checkbox::new(&mut show_behind, "Show behind planet"))
                    .on_hover_text("Show satellites occluded by the planet").changed() {
                        s.hide_behind_earth = !show_behind;
                    }
                }
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_asc_desc_colors, "Asc/Desc colors"))
                    .on_hover_text("Color orbits by ascending/descending node");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.single_color, "Monochrome planes"))
                    .on_hover_text("Use a single color for all orbital planes");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_altitude_lines, "Altitude lines"))
                    .on_hover_text("Draw concentric altitude reference rings");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_torus, "Show torus"))
                    .on_hover_text("Display the orbital torus shell");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_intra_links, "Intra-plane links"))
                    .on_hover_text("Show links between satellites in the same plane");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_links, "Inter-plane links"))
                    .on_hover_text("Show links between satellites in adjacent planes");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_routing_paths, "Show routing paths"))
                    .on_hover_text("Visualize routing algorithms between ground stations");
                ui.indent("routing_opts", |ui| {
                    ui.add_enabled(on && s.show_routing_paths, egui::Checkbox::new(&mut s.show_manhattan_path, "Manhattan (red)"))
                        .on_hover_text("Grid-based hop-by-hop routing path");
                    ui.add_enabled(on && s.show_routing_paths, egui::Checkbox::new(&mut s.show_shortest_path, "Shortest distance (green)"))
                        .on_hover_text("Shortest geometric distance path");
                    ui.add_enabled(on && s.show_routing_paths, egui::Checkbox::new(&mut s.show_radiation_path, "Radiation-aware (cyan)"))
                        .on_hover_text("Path that avoids high-radiation regions");
                    if s.show_radiation_path && s.show_routing_paths {
                        ui.add(egui::Slider::new(&mut s.radiation_weight, 0.0..=10.0).text("Rad weight"))
                            .on_hover_text("Weight of radiation cost in path finding");
                    }
                });
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(on, |ui| {
                        ui.label("Sat:");
                        ui.add(egui::DragValue::new(&mut s.sat_radius).range(1.0..=15.0).speed(0.1))
                            .on_hover_text("Satellite dot radius in pixels");
                        ui.label("Link:");
                        ui.add(egui::DragValue::new(&mut s.link_width).range(0.1..=5.0).speed(0.1))
                            .on_hover_text("ISL link line width in pixels");
                    });
                });
                ui.add_enabled(on, egui::Checkbox::new(&mut s.fixed_sizes, "Fixed sizes (ignore alt)"))
                    .on_hover_text("Keep dot and link sizes constant regardless of altitude");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_sat_border, "Satellite border"))
                    .on_hover_text("Draw an outline ring around each satellite dot");
            });

            ui.label(egui::RichText::new("Body").strong());
            ui.indent("body_opts", |ui| {
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_axes, "Show axes"))
                    .on_hover_text("Display X/Y/Z coordinate axes");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_magnetic_axis, "Show magnetic axis"))
                    .on_hover_text("Show the magnetic dipole axis");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_terminator, "Show sunrise/sunset circle"))
                    .on_hover_text("Draw the day-night terminator line");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_eclipse, "Show eclipsed satellites"))
                    .on_hover_text("Dim satellites in the planet's shadow");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_sun, "Show sun"))
                    .on_hover_text("Display the sun direction indicator");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_polar_circle, "Show polar circle"))
                    .on_hover_text("Draw the Arctic and Antarctic circles");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_equator, "Show equator"))
                    .on_hover_text("Draw the equatorial line");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_graticule, "Show graticule"))
                    .on_hover_text("Draw latitude/longitude grid lines");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_crosshairs, "Show crosshairs"))
                    .on_hover_text("Draw crosshair lines at the view center");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_day_night, "Show day/night"))
                    .on_hover_text("Shade the nightside of the planet");
            });

            ui.label(egui::RichText::new("Radiation").strong());
            ui.indent("radiation_opts", |ui| {
                if let Some(planet) = planets.first_mut() {
                    let rad = &mut planet.radiation;
                    let mut show_belts = s.show_radiation_belts
                        && rad.heatmap_mode == crate::config::HeatmapMode::IgrfRadiation;
                    let mut show_field = s.show_radiation_belts
                        && rad.heatmap_mode == crate::config::HeatmapMode::IgrfField;
                    if ui.add_enabled(on, egui::Checkbox::new(&mut show_belts, "Show radiation belts"))
                    .on_hover_text("Visualize trapped particle radiation belts").changed() {
                        if show_belts {
                            s.show_radiation_belts = true;
                            rad.heatmap_mode = crate::config::HeatmapMode::IgrfRadiation;
                        } else {
                            s.show_radiation_belts = show_field;
                        }
                    }
                    if ui.add_enabled(on, egui::Checkbox::new(&mut show_field, "Show geomagnetic field"))
                    .on_hover_text("Visualize the geomagnetic field strength").changed() {
                        if show_field {
                            s.show_radiation_belts = true;
                            rad.heatmap_mode = crate::config::HeatmapMode::IgrfField;
                        } else {
                            s.show_radiation_belts = show_belts;
                        }
                    }
                    let either = show_belts || show_field;
                    ui.add_enabled(on && either, egui::Checkbox::new(&mut rad.show_heatmap_sphere, "Show heatmap sphere"))
                        .on_hover_text("Render radiation/field data on a sphere");
                    ui.add_enabled(on && either, egui::Checkbox::new(&mut rad.show_sat_exposure, "Color satellites"))
                        .on_hover_text("Color satellites by their radiation exposure");
                    ui.add_enabled_ui(on && either, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Altitude:");
                            ui.add(egui::DragValue::new(&mut rad.heatmap_altitude_km)
                                .range(0.0..=50000.0).speed(50.0).max_decimals(0).suffix(" km"))
                                .on_hover_text("Altitude of the heatmap sphere");
                            if planet.constellations.len() == 1 {
                                if ui.button("Match constellation").on_hover_text("Set altitude to constellation orbit").clicked() {
                                    rad.heatmap_altitude_km = planet.constellations[0].altitude_km;
                                }
                            }
                        });
                    });
                }
            });

            ui.label(egui::RichText::new("Ground").strong());
            ui.indent("ground_opts", |ui| {
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_devices, "Show devices"))
                    .on_hover_text("Display ground-based devices on the surface");
            });

            ui.label(egui::RichText::new("Aesthetics").strong());
            ui.indent("aesthetics_opts", |ui| {
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_clouds, "Show clouds"))
                    .on_hover_text("Overlay cloud layer on the planet");
                ui.add_enabled(on && s.show_day_night, egui::Checkbox::new(&mut s.show_city_lights, "Show city lights"))
                    .on_hover_text("Show city lights on the nightside")
                    .on_disabled_hover_text("Requires Show day/night");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_stars, "Show stars and milky way"))
                    .on_hover_text("Display background star field");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_cities, "City labels"))
                    .on_hover_text("Label major cities on the surface");
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_borders, "Country borders"))
                    .on_hover_text("Draw national border lines");
            });
        });

        let s = &mut tab.settings;
        ui.checkbox(&mut s.show_solar_system, "Show solar system")
            .on_hover_text("Display planets in the solar system");
        ui.indent("solar_system_opts", |ui| {
            let on = s.show_solar_system;
            ui.horizontal(|ui| {
                ui.add_enabled_ui(on, |ui| {
                    ui.label("Scale:");
                    ui.add(egui::DragValue::new(&mut s.solar_system_log_power)
                        .range(0.1..=1.0).speed(0.01).max_decimals(2))
                        .on_hover_text("Log-scale power for distance compression");
                });
            });
            ui.add_enabled_ui(on, |ui| {
                ui.horizontal(|ui| {
                    let label = if self.ss_auto_zoom { "\u{23f8}" } else { "\u{25b6}" };
                    if ui.button(label).on_hover_text("Toggle auto-zoom animation").clicked() {
                        self.ss_auto_zoom = !self.ss_auto_zoom;
                        if self.ss_auto_zoom { self.ss_auto_zoom_time = 0.0; }
                    }
                    ui.label("Auto-zoom");
                    ui.add(egui::DragValue::new(&mut self.ss_auto_zoom_duration).range(5.0..=120.0).speed(0.5).suffix("s"))
                        .on_hover_text("Duration of zoom animation cycle");
                    ui.label("Stay:");
                    ui.add(egui::DragValue::new(&mut self.ss_auto_zoom_stay).range(0.0..=30.0).speed(0.1).suffix("s"))
                        .on_hover_text("Pause duration at each zoom level");
                });
            });
            ui.add_enabled(on, egui::Checkbox::new(&mut s.show_hohmann, "Hohmann transfer"))
                .on_hover_text("Visualize Hohmann transfer orbits between planets");
            let h_on = on && s.show_hohmann;
            let h_sim_time = s.time;
            ui.indent("hohmann_opts", |ui| {
                ui.add_enabled_ui(h_on, |ui| {
                    use crate::solar_system::{HOHMANN_PLANETS, hohmann_transfer_params, next_launch_window_days};
                    let h = &mut self.hohmann;
                    ui.horizontal(|ui| {
                        ui.label("From:");
                        egui::ComboBox::from_id_salt("hohmann_origin")
                            .selected_text(h.origin.label())
                            .show_ui(ui, |ui| {
                                for &body in &HOHMANN_PLANETS {
                                    ui.selectable_value(&mut h.origin, body, body.label());
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.label("To:");
                        egui::ComboBox::from_id_salt("hohmann_dest")
                            .selected_text(h.dest.label())
                            .show_ui(ui, |ui| {
                                for &body in &HOHMANN_PLANETS {
                                    ui.selectable_value(&mut h.dest, body, body.label());
                                }
                            });
                    });
                    if h.origin == h.dest {
                        ui.label("Origin and destination must differ");
                    } else if let Some(params) = hohmann_transfer_params(h.origin, h.dest) {
                        ui.label(format!(
                            "Transfer: {:.1} days ({:.2} yr)",
                            params.transfer_time_days,
                            params.transfer_time_days / 365.25,
                        ));
                        ui.label(format!(
                            "\u{0394}v1: {:.2} km/s  \u{0394}v2: {:.2} km/s",
                            params.departure_dv_km_s,
                            params.arrival_dv_km_s,
                        ));
                        ui.label(format!(
                            "Total \u{0394}v: {:.2} km/s",
                            params.departure_dv_km_s + params.arrival_dv_km_s,
                        ));
                        let ss_ts = self.start_timestamp + Duration::seconds(h_sim_time as i64);
                        let j2000 = ss_ts.signed_duration_since(*crate::solar_system::J2000_EPOCH_PUB).num_seconds() as f64 / 86400.0;
                        let window_wait = next_launch_window_days(h.origin, h.dest, j2000);
                        if let Some(wait) = window_wait {
                            ui.label(format!("Next window: {:.1} days", wait));
                        }
                        if !h.launched {
                            if ui.button("Fast forward and launch").on_hover_text("Skip to next launch window and begin transfer").clicked() {
                                let wait = window_wait.unwrap_or(0.0);
                                s.time += wait * 86400.0;
                                let launch_j2000 = j2000 + wait;
                                if let Some(pos) = crate::solar_system::compute_body_position_au(h.origin, launch_j2000) {
                                    h.departure_angle = pos[1].atan2(pos[0]);
                                }
                                let arrival_j2000 = launch_j2000 + params.transfer_time_days;
                                if let Some(dest_pos) = crate::solar_system::compute_body_position_au(h.dest, arrival_j2000) {
                                    h.arrival_angle = dest_pos[1].atan2(dest_pos[0]);
                                }
                                h.launched = true;
                                h.arrived = false;
                                h.launch_j2000_days = launch_j2000;
                                h.mission_elapsed_days = 0.0;
                                h.trail.clear();
                            }
                        } else {
                            ui.label(format!(
                                "MET: {:.1} days",
                                h.mission_elapsed_days,
                            ));
                            if h.arrived {
                                ui.label("Arrived!");
                            }
                            if ui.button("Reset").on_hover_text("Cancel transfer and reset state").clicked() {
                                h.launched = false;
                                h.arrived = false;
                                h.mission_elapsed_days = 0.0;
                                h.trail.clear();
                            }
                        }
                    }
                });
            });
        });

        ui.checkbox(&mut self.show_planet_sizes, "Show planet sizes")
            .on_hover_text("Display relative planet sizes");
        ui.indent("planet_sizes_opts", |ui| {
            ui.add_enabled_ui(self.show_planet_sizes, |ui| {
                ui.horizontal(|ui| {
                    let label = if self.planet_sizes_auto_zoom { "\u{23f8}" } else { "\u{25b6}" };
                    if ui.button(label).on_hover_text("Toggle auto-zoom animation").clicked() {
                        self.planet_sizes_auto_zoom = !self.planet_sizes_auto_zoom;
                        if self.planet_sizes_auto_zoom { self.planet_sizes_auto_time = 0.0; }
                    }
                    ui.label("Auto-zoom");
                    ui.add(egui::DragValue::new(&mut self.planet_sizes_zoom_duration).range(5.0..=120.0).speed(0.5).suffix("s"))
                        .on_hover_text("Duration of zoom animation cycle");
                    ui.label("Stay:");
                    ui.add(egui::DragValue::new(&mut self.planet_sizes_stay_duration).range(0.0..=30.0).speed(0.1).suffix("s"))
                        .on_hover_text("Pause duration at each zoom level");
                });
            });
        });
        ui.checkbox(&mut self.auto_hide_tab_bar, "Auto-hide UI")
            .on_hover_text("Hide the tab bar and sidebar automatically");
        ui.checkbox(&mut self.auto_cycle_tabs, "Auto-cycle tabs")
            .on_hover_text("Cycle through tabs automatically");
        ui.indent("cycle_opts", |ui| {
            ui.add_enabled_ui(self.auto_cycle_tabs, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Interval:");
                    ui.add(egui::DragValue::new(&mut self.cycle_interval).range(1.0..=60.0).speed(0.5).suffix("s"))
                        .on_hover_text("Time between automatic tab switches");
                });
            });
        });

        let prev_slideshow = self.slideshow_mode;
        ui.checkbox(&mut self.slideshow_mode, "Slideshow mode")
            .on_hover_text("Full-screen presentation with fade transitions");
        if self.slideshow_mode && !prev_slideshow {
            self.auto_cycle_tabs = true;
            self.auto_hide_tab_bar = true;
        } else if !self.slideshow_mode && prev_slideshow {
            self.auto_cycle_tabs = false;
            self.auto_hide_tab_bar = false;
        }

        ui.separator();
        ui.label(egui::RichText::new("Rendering").strong());
        ui.checkbox(&mut self.dark_mode, "Dark mode")
            .on_hover_text("Use dark background theme");
        ui.horizontal(|ui| {
            ui.label("Texture:");
            egui::ComboBox::from_id_salt("tex_res")
                .selected_text(self.texture_resolution.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.texture_resolution, TextureResolution::R512, "512");
                    ui.selectable_value(&mut self.texture_resolution, TextureResolution::R1024, "1K");
                    ui.selectable_value(&mut self.texture_resolution, TextureResolution::R2048, "2K");
                    ui.selectable_value(&mut self.texture_resolution, TextureResolution::R8192, "8K");
                    #[cfg(not(target_arch = "wasm32"))]
                    ui.selectable_value(&mut self.texture_resolution, TextureResolution::R21504, "21K");
                });
        });
        ui.checkbox(&mut self.use_gpu_rendering, "GPU rendering")
            .on_hover_text("Use GPU shaders for planet rendering");
        #[cfg(not(target_arch = "wasm32"))]
        ui.checkbox(&mut self.tile_overlay.enabled, "Satellite tiles (Esri)")
            .on_hover_text("Overlay Esri satellite imagery tiles");
    }
}
