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

## TODO

- Ability to see other planets around the one currently being viewed. When 
zoomed out, we should be able to see the solar system.
- Clicking another planet should make that planet be focused instead.
- Camera tilt (Ctrl+drag) with perspective projection for viewing satellites from the side

## Image Credits

Planet textures are from:

**[Solar System Scope](https://www.solarsystemscope.com/textures/)** (CC BY 4.0)
- Earth, Moon, Mars, Mercury, Venus, Jupiter, Saturn, Sun (2K, 8K)
- Stars, Stars + Milky Way (2K, 8K)

**[Solar System in 16K Texture Pack](https://www.artstation.com/marketplace/p/5oNl/solar-system-in-16k-texture-pack)** by Textures for Planets
- Earth (21K diffuse, 16K clouds, normal, specular, cities)
- Mercury (16K diffuse, normal, specular, roughness, displacement)
- Venus (16K diffuse, normal, roughness, displacement)

**[Steve Albers' Planetary Maps](https://stevealbers.net/albers/sos/sos.html)**
- Ganymede (4K, Bjorn Jonsson modified)
- Io (2K, Voyager + Galileo composite)
- Europa (4K, Juno data)
- Charon (8K)
- Enceladus (8K, Cassini)

**[Askaniy on DeviantArt](https://www.deviantart.com/askaniy)** (CC BY 3.0)
- Titan True Color Map (2K, Cassini infrared + Huygens data)
- Triton Texture Map (2K, Voyager 2 calibrated color)
- Phobos Texture Map (2K, Mars Express)

**[Oleg-Pluton on DeviantArt](https://www.deviantart.com/oleg-pluton)**
- Mimas Texture Map (2K, Cassini)

**[FarGetaNik on DeviantArt](https://www.deviantart.com/fargetanik)**
- Iapetus Truecolor Texture Map (2K, Cassini)

**[Bjorn Jonsson's Planetary Maps](https://bjj.mmedia.is/)**
- Callisto (4K)

**[Celestia Project](https://github.com/CelestiaProject/CelestiaContent)** (GPL)
- Vesta (4K, Dawn mission mosaic)

**[NASA Science Assets](https://assets.science.nasa.gov/)**
- Additional planetary imagery

## TLE Data

Real-time satellite tracking data is fetched from
[CelesTrak](https://celestrak.org/) maintained by Dr. T.S. Kelso.
- ISS, Starlink, OneWeb, Kuiper
- Iridium, Iridium NEXT, Globalstar, Orbcomm
- GPS, Galileo, GLONASS, Beidou
- Molniya, Planet
- ...
