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
        let s = &mut self.tabs[active].settings;
        let (time_ref, rotation_ref, zoom_ref, speed_ref, animate_ref, earth_fixed_ref, follow_sat_ref, show_cam_ref) =
            (&mut s.time, &mut s.rotation, &mut s.zoom, &mut s.speed, &mut s.animate, &mut s.earth_fixed_camera, &mut s.follow_satellite, &mut s.show_camera_windows);

        let body_rotation = body_rotation_angle(current_body, *time_ref, self.current_gmst);

        ui.label(egui::RichText::new("Camera").strong());
        let (lat, base_lon) = matrix_to_lat_lon(rotation_ref);
        let geo_lon = if *earth_fixed_ref {
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
            ui.label("Alt:");
            let mut alt_km = 10000.0 / *zoom_ref;
            if ui.add(egui::DragValue::new(&mut alt_km).range(0.5..=1000000.0).speed(100.0).suffix(" km")).changed() {
                *zoom_ref = (10000.0 / alt_km).clamp(0.01, 20000.0);
            }
            lat_deg = lat_deg.clamp(-90.0, 90.0);
            while lon_deg > 180.0 { lon_deg -= 360.0; }
            while lon_deg < -180.0 { lon_deg += 360.0; }
            if lat_changed || lon_changed {
                let target_lon = if *earth_fixed_ref {
                    lon_deg.to_radians()
                } else {
                    lon_deg.to_radians() + body_rotation
                };
                *rotation_ref = lat_lon_to_matrix(lat_deg.to_radians(), target_lon);
            }
        });
        let was_earth_fixed = *earth_fixed_ref;
        ui.checkbox(earth_fixed_ref, "Fixed Lat/Lon");
        if *earth_fixed_ref != was_earth_fixed {
            let cos_a = body_rotation.cos();
            let sin_a = body_rotation.sin();
            let body_y_rot = Matrix3::new(
                cos_a, 0.0, sin_a,
                0.0, 1.0, 0.0,
                -sin_a, 0.0, cos_a,
            );
            if *earth_fixed_ref {
                *rotation_ref *= body_y_rot;
            } else {
                *rotation_ref *= body_y_rot.transpose();
            }
        }
        ui.horizontal(|ui| {
            for (label, lat, lon) in [("N", 90.0_f64, 0.0_f64), ("S", -90.0, 0.0), ("E", 0.0, 90.0), ("W", 0.0, -90.0)] {
                if ui.button(label).clicked() {
                    let target_lon = if *earth_fixed_ref {
                        lon.to_radians()
                    } else {
                        lon.to_radians() + body_rotation
                    };
                    *rotation_ref = lat_lon_to_matrix(lat.to_radians(), target_lon);
                }
            }
        });

        ui.checkbox(follow_sat_ref, "Follow satellite");
        ui.checkbox(show_cam_ref, "Show camera windows");

        ui.separator();
        ui.label(egui::RichText::new("Simulation").strong());
        ui.horizontal(|ui| {
            ui.label("Speed:");
            let abs_speed = speed_ref.abs();
            let drag_speed = if abs_speed > 31_536_000.0 { 100_000.0 } else if abs_speed > 86400.0 { 1000.0 } else if abs_speed > 3600.0 { 100.0 } else { 1.0 };
            ui.add(egui::DragValue::new(speed_ref).range(-3_153_600_000.0..=3_153_600_000.0).speed(drag_speed));
            if ui.button("⏪").clicked() {
                *speed_ref = -*speed_ref;
            }
            let pause_label = if *animate_ref { "⏸" } else { "▶" };
            if ui.button(pause_label).clicked() {
                *animate_ref = !*animate_ref;
            }
            if abs_speed > 60.0 {
                let label = if abs_speed >= 31_536_000.0 {
                    format!("{:.1} earth years/s", abs_speed / 31_536_000.0)
                } else if abs_speed >= 2_592_000.0 {
                    format!("{:.1} earth months/s", abs_speed / 2_592_000.0)
                } else if abs_speed >= 86400.0 {
                    format!("{:.1} earth days/s", abs_speed / 86400.0)
                } else if abs_speed >= 3600.0 {
                    format!("{:.1} earth hours/s", abs_speed / 3600.0)
                } else {
                    format!("{:.1} earth minutes/s", abs_speed / 60.0)
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
        if ui.button("Sync time").clicked() {
            *time_ref = self.real_time;
        }

        {
            let s = &mut self.tabs[active].settings;

            ui.separator();
            ui.label(egui::RichText::new("Display").strong());

            ui.checkbox(&mut s.render_planet, "Show planet");
            ui.indent("planet_opts", |ui| {
                let on = s.render_planet;
                {
                    let mut show_behind = !s.hide_behind_earth;
                    if ui.add_enabled(on, egui::Checkbox::new(&mut show_behind, "Show behind planet")).changed() {
                        s.hide_behind_earth = !show_behind;
                    }
                }
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_clouds, "Show clouds"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_day_night, "Show day/night"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_terminator, "Show sunrise/sunset circle"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_orbits, "Show orbits"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_coverage, "Show coverage"));
                ui.indent("coverage_opts", |ui| {
                    ui.add_enabled_ui(on && s.show_coverage, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Angle:");
                            ui.add(egui::DragValue::new(&mut s.coverage_angle)
                                .range(0.5..=70.0).speed(0.1).max_decimals(1).suffix("°"));
                        });
                    });
                });
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_altitude_lines, "Altitude lines"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_axes, "Show axes"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_polar_circle, "Show polar circle"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_equator, "Show equator"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_borders, "Country borders"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_cities, "City labels"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_devices, "Show devices"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_stars, "Show stars and milky way"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_torus, "Show torus"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_asc_desc_colors, "Asc/Desc colors"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.single_color, "Monochrome planes"));
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(on, |ui| {
                        ui.label("Sat:");
                        ui.add(egui::DragValue::new(&mut s.sat_radius).range(1.0..=15.0).speed(0.1));
                        ui.label("Link:");
                        ui.add(egui::DragValue::new(&mut s.link_width).range(0.1..=5.0).speed(0.1));
                    });
                });
                ui.add_enabled(on, egui::Checkbox::new(&mut s.fixed_sizes, "Fixed sizes (ignore alt)"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_intra_links, "Intra-plane links"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_links, "Inter-plane links"));
                ui.add_enabled(on, egui::Checkbox::new(&mut s.show_routing_paths, "Show routing paths"));
                ui.indent("routing_opts", |ui| {
                    ui.add_enabled(on && s.show_routing_paths, egui::Checkbox::new(&mut s.show_manhattan_path, "Manhattan (red)"));
                    ui.add_enabled(on && s.show_routing_paths, egui::Checkbox::new(&mut s.show_shortest_path, "Shortest distance (green)"));
                });
            });

            ui.checkbox(&mut s.show_solar_system, "Show solar system");
            ui.indent("solar_system_opts", |ui| {
                let on = s.show_solar_system;
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(on, |ui| {
                        ui.label("Scale:");
                        ui.add(egui::DragValue::new(&mut s.solar_system_log_power)
                            .range(0.1..=1.0).speed(0.01).max_decimals(2));
                    });
                });
                ui.add_enabled_ui(on, |ui| {
                    ui.horizontal(|ui| {
                        let label = if self.ss_auto_zoom { "\u{23f8}" } else { "\u{25b6}" };
                        if ui.button(label).clicked() {
                            self.ss_auto_zoom = !self.ss_auto_zoom;
                            if self.ss_auto_zoom { self.ss_auto_zoom_time = 0.0; }
                        }
                        ui.label("Auto-zoom");
                        ui.add(egui::DragValue::new(&mut self.ss_auto_zoom_duration).range(5.0..=120.0).speed(0.5).suffix("s"));
                        ui.label("Stay:");
                        ui.add(egui::DragValue::new(&mut self.ss_auto_zoom_stay).range(0.0..=30.0).speed(0.1).suffix("s"));
                    });
                });
            });

            ui.checkbox(&mut self.show_planet_sizes, "Show planet sizes");
            ui.indent("planet_sizes_opts", |ui| {
                ui.add_enabled_ui(self.show_planet_sizes, |ui| {
                    ui.horizontal(|ui| {
                        let label = if self.planet_sizes_auto_zoom { "\u{23f8}" } else { "\u{25b6}" };
                        if ui.button(label).clicked() {
                            self.planet_sizes_auto_zoom = !self.planet_sizes_auto_zoom;
                            if self.planet_sizes_auto_zoom { self.planet_sizes_auto_time = 0.0; }
                        }
                        ui.label("Auto-zoom");
                        ui.add(egui::DragValue::new(&mut self.planet_sizes_zoom_duration).range(5.0..=120.0).speed(0.5).suffix("s"));
                        ui.label("Stay:");
                        ui.add(egui::DragValue::new(&mut self.planet_sizes_stay_duration).range(0.0..=30.0).speed(0.1).suffix("s"));
                    });
                });
            });
            ui.checkbox(&mut s.show_ground_track, "Show ground track");

            ui.checkbox(&mut self.auto_hide_tab_bar, "Auto-hide UI");
            ui.checkbox(&mut self.auto_cycle_tabs, "Auto-cycle tabs");
            ui.indent("cycle_opts", |ui| {
                ui.add_enabled_ui(self.auto_cycle_tabs, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Interval:");
                        ui.add(egui::DragValue::new(&mut self.cycle_interval).range(1.0..=60.0).speed(0.5).suffix("s"));
                    });
                });
            });

            ui.separator();
            ui.label(egui::RichText::new("Rendering").strong());
            ui.checkbox(&mut self.dark_mode, "Dark mode");
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
            ui.checkbox(&mut self.use_gpu_rendering, "GPU rendering");
            #[cfg(not(target_arch = "wasm32"))]
            ui.checkbox(&mut self.tile_overlay.enabled, "Satellite tiles (Esri)");
        }
    }
}
