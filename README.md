# LEO Constellation Visualizer

A satellite constellation visualization tool built with Rust and egui.

## Features

- Walker Delta and Star constellation configurations
- Real-time TLE satellite tracking (Starlink, OneWeb, Iridium, GPS, etc.)
- Multi-planet support (Earth, Mars, Moon, and other solar system bodies)
- 3D globe view with GPU-accelerated rendering
- Torus topology view for constellation analysis
- Ground track visualization
- Inter-satellite link routing visualization
- Coverage analysis

## Building

```bash
cargo build --release
```

## Image Credits

Planet textures are from:

**[Solar System Scope](https://www.solarsystemscope.com/textures/)** (CC BY 4.0)
- Earth, Moon, Mars, Mercury, Venus, Jupiter, Saturn, Sun (2K, 8K)

**[Solar System in 16K Texture Pack](https://www.artstation.com/marketplace/p/5oNl/solar-system-in-16k-texture-pack)** by Textures for Planets
- Earth (21K diffuse, 16K clouds, normal, specular, cities)
- Mercury (16K diffuse, normal, specular, roughness, displacement)
- Venus (16K diffuse, normal, roughness, displacement)

**[NASA Science Assets](https://assets.science.nasa.gov/)**
- Additional planetary imagery

## TLE Data

Real-time satellite tracking data is fetched from
[CelesTrak](https://celestrak.org/) maintained by Dr. T.S. Kelso.
