//! Demo mode setup with pre-configured multi-tab constellation layouts.

use crate::celestial::CelestialBody;
use crate::config::{ConstellationConfig, PlanetConfig, Preset, TabConfig};
use crate::tle::{TlePreset, TleLoadState};
use crate::walker::WalkerType;
use crate::App;
use egui_dock::DockState;

impl App {
    pub(crate) fn setup_demo(&mut self) {
        let v = &mut self.viewer;
        v.tabs.clear();
        v.tab_counter = 0;

        let leo_tle = [
            TlePreset::Starlink, TlePreset::OneWeb, TlePreset::Kuiper, TlePreset::Iridium,
            TlePreset::IridiumNext, TlePreset::Globalstar, TlePreset::Orbcomm,
        ];
        let geo_tle = [
            TlePreset::Gps, TlePreset::Galileo, TlePreset::Glonass, TlePreset::Beidou,
            TlePreset::Molniya, TlePreset::Planet,
        ];

        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Inclination: 90° vs 60°".to_string());
            for (inc, label) in [(90.0, "90°"), (60.0, "60°")] {
                tab.planet_counter += 1;
                let mut planet = PlanetConfig::new(format!("Earth ({})", label));
                planet.celestial_body = CelestialBody::Earth;
                let mut cons = ConstellationConfig::new(0);
                cons.sats_per_plane = 11;
                cons.num_planes = 6;
                cons.inclination = inc;
                cons.altitude_km = 780.0;
                cons.walker_type = WalkerType::Star;
                planet.constellations.push(cons);
                tab.planets.push(planet);
            }
            v.tabs.push(tab);
        }

        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Walker: Star vs Delta".to_string());
            for (wt, label) in [(WalkerType::Star, "Star"), (WalkerType::Delta, "Delta")] {
                tab.planet_counter += 1;
                let mut planet = PlanetConfig::new(format!("Mars ({})", label));
                planet.celestial_body = CelestialBody::Mars;
                let mut cons = ConstellationConfig::new(0);
                cons.sats_per_plane = 8;
                cons.num_planes = 4;
                cons.inclination = 70.0;
                cons.altitude_km = 500.0;
                cons.walker_type = wt;
                planet.constellations.push(cons);
                tab.planets.push(planet);
            }
            v.tabs.push(tab);
        }

        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Phasing: F=0 vs F=2".to_string());
            for (f, label) in [(0.0, "F=0"), (2.0, "F=2")] {
                tab.planet_counter += 1;
                let mut planet = PlanetConfig::new(format!("Venus ({})", label));
                planet.celestial_body = CelestialBody::Venus;
                let mut cons = ConstellationConfig::new(0);
                cons.sats_per_plane = 6;
                cons.num_planes = 6;
                cons.inclination = 80.0;
                cons.altitude_km = 400.0;
                cons.phasing = f;
                planet.constellations.push(cons);
                tab.planets.push(planet);
            }
            v.tabs.push(tab);
        }

        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Altitude: VLEO/LEO/MEO/GEO".to_string());
            tab.planet_counter += 1;
            let mut planet = PlanetConfig::new("Mercury".to_string());
            planet.celestial_body = CelestialBody::Mercury;
            let altitudes = [(200.0, 0), (550.0, 1), (8000.0, 2), (35786.0, 3)];
            for (alt, color) in altitudes {
                let mut cons = ConstellationConfig::new(color);
                cons.sats_per_plane = 1;
                cons.num_planes = 1;
                cons.inclination = 0.0;
                cons.altitude_km = alt;
                planet.constellations.push(cons);
            }
            tab.planets.push(planet);
            v.tabs.push(tab);
        }

        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Real: LEO Constellations".to_string());
            tab.planet_counter += 1;
            let mut planet = PlanetConfig::new("Earth".to_string());
            planet.celestial_body = CelestialBody::Earth;
            for preset in leo_tle {
                planet.tle_selections.insert(preset, (true, TleLoadState::NotLoaded, None));
            }
            tab.planets.push(planet);
            v.tabs.push(tab);
        }

        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Real: LEO + Navigation".to_string());
            tab.planet_counter += 1;
            let mut planet = PlanetConfig::new("Earth".to_string());
            planet.celestial_body = CelestialBody::Earth;
            for preset in leo_tle.iter().chain(geo_tle.iter()) {
                planet.tle_selections.insert(*preset, (true, TleLoadState::NotLoaded, None));
            }
            tab.planets.push(planet);
            v.tabs.push(tab);
        }

        {
            v.tab_counter += 1;
            let mut tab = TabConfig::new_empty("Starlink: Simulated vs Real".to_string());
            tab.planet_counter += 1;
            let mut planet_sim = PlanetConfig::new("Simulated".to_string());
            planet_sim.celestial_body = CelestialBody::Earth;
            let mut cons = ConstellationConfig::new(0);
            cons.preset = Preset::Starlink;
            cons.sats_per_plane = 22;
            cons.num_planes = 72;
            cons.inclination = 53.0;
            cons.altitude_km = 550.0;
            cons.walker_type = WalkerType::Delta;
            planet_sim.constellations.push(cons);
            tab.planets.push(planet_sim);
            tab.planet_counter += 1;
            let mut planet_real = PlanetConfig::new("Real TLE".to_string());
            planet_real.celestial_body = CelestialBody::Earth;
            planet_real.tle_selections.insert(TlePreset::Starlink, (true, TleLoadState::NotLoaded, None));
            tab.planets.push(planet_real);
            v.tabs.push(tab);
        }

        self.dock_state = DockState::new(vec![0]);
        for i in 1..v.tabs.len() {
            self.dock_state.push_to_focused_leaf(i);
        }

        v.auto_cycle_tabs = true;
        v.cycle_interval = 8.0;
        v.last_cycle_time = 0.0;
    }
}
