
## TODO

### Physics / Orbital Mechanics
- Solar radiation pressure perturbation (dominant non-gravitational force for high area-to-mass satellites)
- Differential J2: show relative RAAN drift between shells at different altitudes (°/year stat)
- Delta-v / station-keeping budget: annual m/s needed to maintain orbit against drag at given altitude + ballistic coefficient
- HCW relative motion frame for close-formation visualization

### Constellation Features
- Mass/power satellite attributes (kg, kW, solar panel area) for trade studies and power generation estimates
- Optical ISL link budget: bandwidth (Gbps) per link based on distance using Friis equation, color-coded throughput
- Altitude randomization: distribute satellites between min..max altitude to simulate realistic orbital variation

### Demo Tabs
1. **Coverage**: Increase to 20×20 satellites per constellation
2. **Eccentricity**: Remove circular, only Molniya. 20 sats × 10 planes. Speed 250×
3. **SSO**: Justify the demo visually — orbit plane stays aligned with terminator
4. **ISL Topology**: Hide orbits, show only ISL links. Increase to 20×20
5. **Starlink shells**: Verify realistic sats-per-shell counts (currently correct per SpaceX filings)
6. **Drag & Decay**: Show 1 satellite, 1 orbit, full orbit camera view, 5000× speed, live altitude display
7. **Kessler**: Debris in different color, same size as satellites
8. **Keplerian vs J2**: Increase speed to 2000×, lower inclination for faster visible RAAN drift
9. **Radiation**: Show at 10,000 km alt. Fix both planets showing geomagnetic only — first should show radiation belts
10. **Starlink**: All shells ~540-570 km (correct — differentiated by inclination, not altitude)
11. **IRIS²**: Show at 18,000 km alt zoom
12. **Solar System / Planet Sizes**: Investigate texture loading lag spikes (not caching across frames?)
13. **TLE prefetch**: Investigate why TLEs aren't loading on demo init

### Other
- LLM tool calls
- WASM
