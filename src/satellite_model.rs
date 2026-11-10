//! Import self-contained glTF spacecraft assets without altering their source files.
use base64::Engine as _;
use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};
use std::collections::{BTreeMap, HashMap};

#[path = "satellite_parts.rs"]
mod catalogue;
use catalogue::PartInfo;
#[path = "satellite_facts.rs"]
mod facts;
pub(crate) use facts::Facts;

#[derive(Clone, Copy)]
enum MeshScale {
    Axis(&'static str),
    AssemblyAxis(&'static [&'static str]),
    Diagonal(&'static str),
}

pub(crate) struct Asset {
    pub name: &'static str,
    pub path: &'static str,
    pub credit: &'static str,
    pub source: &'static str,
    pub documentation: &'static str,
    pub reference_dimensions: [f32; 3],
    pub dimension_label: &'static str,
    scale_mesh: Option<MeshScale>,
    pub dimension_source: &'static str,
    pub facts: &'static facts::Facts,
    catalogue: &'static [catalogue::Spec],
    unassigned: &'static PartInfo,
}

impl Asset {
    pub fn load(&self, bytes: &[u8]) -> Result<Model, String> {
        let mut model = Model::load(bytes)?;
        if self.name == "SMAP" {
            for material in &mut model.materials {
                material.appearance = match material.name.as_str() {
                    "antenna_a" => MaterialAppearance::GoldReflector,
                    "antenna_support_structure" => MaterialAppearance::ReflectorRim,
                    _ => MaterialAppearance::Original,
                };
            }
        }
        if matches!(self.name, "SMAP" | "SWOT" | "GRACE-FO") {
            for part in &mut model.parts {
                if part.name == "_root"
                    && matches!(
                        model.materials[part.material].name.as_str(),
                        "clear" | "transparent"
                    )
                {
                    part.inspectable = false;
                }
            }
        }
        if self.reference_span(&model) <= 0.0 {
            return Err("The model has no reviewed physical scale reference.".into());
        }
        let regions = match self.name {
            "Aqua" => catalogue::AQUA_REGIONS,
            "SWOT" => catalogue::SWOT_REGIONS,
            "GRACE-FO" => catalogue::GRACEFO_REGIONS,
            "GPM Core Observatory" => catalogue::GPM_REGIONS,
            "Suomi NPP" => catalogue::SNPP_REGIONS,
            _ => &[],
        };
        model.partition_surfaces(regions, self.catalogue);
        let mut assigned: Vec<_> = model
            .parts
            .iter()
            .map(|p| !p.is_visible(&model.materials))
            .collect();
        for spec in self.catalogue {
            let meshes: Vec<_> = model
                .parts
                .iter()
                .enumerate()
                .filter_map(|(index, part)| {
                    (!assigned[index]
                        && (spec.meshes.iter().any(|name| {
                            name.strip_suffix('*')
                                .map_or(part.name == *name, |prefix| part.name.starts_with(prefix))
                        }) || spec
                            .materials
                            .contains(&model.materials[part.material].name.as_str())))
                    .then_some(index)
                })
                .collect();
            if meshes.is_empty() {
                continue;
            }
            for &index in &meshes {
                assigned[index] = true;
            }
            model.add_component(&spec.info, meshes);
        }
        for (region, annotation) in regions.iter().enumerate() {
            let meshes: Vec<_> = model
                .parts
                .iter()
                .enumerate()
                .filter_map(|(i, p)| (p.region == Some(region) && !assigned[i]).then_some(i))
                .collect();
            for &index in &meshes {
                assigned[index] = true;
            }
            if !meshes.is_empty() {
                model.add_component(&annotation.info, meshes);
            }
        }
        let remaining: Vec<_> = assigned
            .iter()
            .enumerate()
            .filter_map(|(i, &used)| (!used).then_some(i))
            .collect();
        if !remaining.is_empty() {
            model.add_component(self.unassigned, remaining);
        }
        for vertex in &mut model.vertices {
            let part = &model.parts[vertex.part as usize];
            vertex.part = part.component as u32;
            vertex.offset = part.offset.into();
        }
        Ok(model)
    }

    pub fn metres_per_unit(&self, model: &Model) -> f32 {
        let published_span = self
            .reference_dimensions
            .into_iter()
            .fold(0.0_f32, f32::max);
        published_span / self.reference_span(model)
    }

    fn reference_span(&self, model: &Model) -> f32 {
        match self.scale_mesh {
            Some(scale) => {
                let diagonal = matches!(scale, MeshScale::Diagonal(_));
                let mut parts = model.parts.iter().filter(|p| match scale {
                    MeshScale::Axis(name) | MeshScale::Diagonal(name) => p.name == name,
                    MeshScale::AssemblyAxis(names) => names.contains(&p.name.as_str()),
                });
                let Some(first) = parts.next() else {
                    return 0.0;
                };
                let (min, max) = parts.fold((first.min, first.max), |(min, max), part| {
                    (min.inf(&part.min), max.sup(&part.max))
                });
                if diagonal {
                    (max - min).norm()
                } else {
                    (max - min).max()
                }
            }
            None => model.dimensions.max() / model.radius,
        }
    }
}

pub(crate) const ASSETS: [Asset; 13] = [
    Asset {
        name: "Sentinel-6",
        path: "assets/satellite-sentinel6.glb",
        credit: "NASA/Matthew J. Garcia",
        source: "https://science.nasa.gov/3d-resources/jason-continuity-of-service-sentinel-6/",
        documentation: "https://sentinels.copernicus.eu/missions/sentinel-6/satellite-description",
        reference_dimensions: [5.13, 4.17, 2.34],
        dimension_label: "Deployed",
        scale_mesh: None,
        facts: &facts::SENTINEL6,
        dimension_source:
            "https://www-cdn.eumetsat.int/files/2020-07/sentinel6-4pager_factsheet-v09.pdf",
        catalogue: catalogue::SENTINEL6,
        unassigned: &catalogue::SENTINEL_OTHER,
    },
    Asset {
        name: "Aqua",
        path: "assets/satellite-aqua.glb",
        credit: "NASA/Kevin K. Lane",
        source: "https://science.nasa.gov/3d-resources/aqua-a/",
        documentation:
            "https://science.nasa.gov/mission/aqua/about-aqua-spacecraft-and-instruments/",
        reference_dimensions: [4.8, 16.7, 8.0],
        dimension_label: "Deployed",
        scale_mesh: None,
        facts: &facts::AQUA,
        dimension_source:
            "https://science.nasa.gov/mission/aqua/about-aqua-spacecraft-and-instruments/",
        catalogue: catalogue::AQUA,
        unassigned: &catalogue::AQUA_OTHER,
    },
    Asset {
        name: "Landsat 8", path: "assets/satellite-landsat8.glb",
        credit: "NASA/Christopher R. Meaney",
        source: "https://science.nasa.gov/3d-resources/landsat-8/",
        documentation: "https://science.nasa.gov/mission/landsat-8/",
        reference_dimensions: [3.0,2.4,2.4], dimension_label: "Bus length / diameter", scale_mesh: Some(MeshScale::Axis("Layer_0 / 15")),
        dimension_source: "https://www.usgs.gov/landsat-missions/landsat-8",
        facts: &facts::LANDSAT8, catalogue: catalogue::LANDSAT8, unassigned: &catalogue::LANDSAT8_OTHER,
    },
    Asset {
        name: "ICESat-2", path: "assets/satellite-icesat2.glb",
        credit: "NASA/Brian E. Kumanchik",
        source: "https://science.nasa.gov/3d-resources/ice-clouds-and-land-elevation-satellite-2-icesat-2-b/",
        documentation: "https://science.nasa.gov/mission/icesat-2/",
        reference_dimensions: [3.8,2.5,10.0], dimension_label: "Deployed (approx.)", scale_mesh: None,
        dimension_source: "https://icesat-2.gsfc.nasa.gov/sites/default/files/page_files/FactSheet_0828_2018_0.pdf",
        facts: &facts::ICESAT2, catalogue: catalogue::ICESAT2, unassigned: &catalogue::ICESAT2_OTHER,
    },
    Asset {
        name: "SMAP", path: "assets/satellite-smap.glb",
        credit: "NASA/JPL-Caltech/VTAD",
        source: "https://smap.jpl.nasa.gov/resources/99/smap-3d-model/",
        documentation: "https://science.nasa.gov/mission/smap/",
        reference_dimensions: [9.7,7.1,6.8], dimension_label: "Deployed", scale_mesh: None,
        dimension_source: "https://www.jpl.nasa.gov/news/press_kits/smaplaunch.pdf",
        facts: &facts::SMAP, catalogue: catalogue::SMAP, unassigned: &catalogue::SMAP_OTHER,
    },
    Asset {
        name: "MiRaTa", path: "assets/satellite-mirata.glb",
        credit: "NASA/Christopher R. Meaney",
        source: "https://science.nasa.gov/3d-resources/cubesat-mirata/",
        documentation: "https://www.nasa.gov/wp-content/uploads/2016/07/elana_14_jpss1_cubesat_factsheet_508.pdf",
        reference_dimensions: [0.10,0.10,0.34], dimension_label: "Stowed body (3U)", scale_mesh: None,
        dimension_source: "https://www.haystack.mit.edu/wp-content/uploads/2020/07/conf_NEROC2016_cahoy_web.pdf",
        facts: &facts::MIRATA, catalogue: catalogue::MIRATA, unassigned: &catalogue::MIRATA_OTHER,
    },
    Asset {
        name: "GOES-R series", path: "assets/satellite-goesr.glb",
        credit: "NASA/NOAA",
        source: "https://science.nasa.gov/mission/goes-spacecraft-3d-model/",
        documentation: "https://www.nesdis.noaa.gov/our-satellites/currently-flying/goes-east-west/goes-r-series-spacecraft-instruments",
        reference_dimensions: [8.5,0.0,0.0], dimension_label: "Magnetometer boom", scale_mesh: Some(MeshScale::Diagonal("Boom")),
        dimension_source: "https://repository.library.noaa.gov/view/noaa/61685/noaa_61685_DS1.pdf",
        facts: &facts::GOESR, catalogue: catalogue::GOESR, unassigned: &catalogue::GOESR_OTHER,
    },
    Asset {
        name: "SWOT", path: "assets/satellite-swot.glb",
        credit: "NASA/JPL-Caltech",
        source: "https://swot.jpl.nasa.gov/resources/86/swot-3d-model/",
        documentation: "https://www.jpl.nasa.gov/press-kits/swot/mission-overview/",
        reference_dimensions: [14.875,0.0,0.0], dimension_label: "Solar-array span", scale_mesh: Some(MeshScale::Axis("shiny_panel")),
        dimension_source: "https://www.jpl.nasa.gov/press-kits/swot/quick-facts/",
        facts: &facts::SWOT, catalogue: &[], unassigned: &catalogue::SWOT_OTHER,
    },
    Asset {
        name: "GRACE-FO", path: "assets/satellite-gracefo.glb",
        credit: "NASA/JPL-Caltech/VTAD",
        source: "https://gracefo.jpl.nasa.gov/resources/86/grace-fo-3d-model/",
        documentation: "https://www.jpl.nasa.gov/news/press_kits/grace-fo/mission/spacecraft/",
        reference_dimensions: [3.123,1.943,0.78], dimension_label: "Body", scale_mesh: Some(MeshScale::Axis("BUS")),
        dimension_source: "https://www.jpl.nasa.gov/news/press_kits/grace-fo/facts/",
        facts: &facts::GRACEFO, catalogue: catalogue::GRACEFO, unassigned: &catalogue::GRACEFO_OTHER,
    },
    Asset {
        name: "OCO-2", path: "assets/satellite-oco2.glb",
        credit: "NASA/Christopher R. Meaney",
        source: "https://science.nasa.gov/3d-resources/orbiting-carbon-observatory-oco-2/",
        documentation: "https://ocov2.jpl.nasa.gov/observatory/spacecraft/",
        reference_dimensions: [9.0,0.0,0.0], dimension_label: "Solar-array span (approx.)", scale_mesh: None,
        dimension_source: "https://www.jpl.nasa.gov/news/press_kits/oco2-launch-press-kit.pdf",
        facts: &facts::OCO2, catalogue: catalogue::OCO2, unassigned: &catalogue::OCO2_OTHER,
    },
    Asset {
        name: "Aura", path: "assets/satellite-aura.glb",
        credit: "NASA/Christopher R. Meaney",
        source: "https://science.nasa.gov/3d-resources/aura-d/",
        documentation: "https://science.nasa.gov/mission/aura/spacecraft-and-instruments/",
        reference_dimensions: [4.70,17.37,6.91], dimension_label: "Deployed", scale_mesh: None,
        dimension_source: "https://web.archive.org/web/20170701000000/http://www.nasa.gov/pdf/60914main_AuraPressKit6-30.pdf",
        facts: &facts::AURA, catalogue: catalogue::AURA, unassigned: &catalogue::AURA_OTHER,
    },
    Asset {
        name: "GPM Core Observatory", path: "assets/satellite-gpm.glb",
        credit: "NASA/Christopher R. Meaney",
        source: "https://science.nasa.gov/3d-resources/global-precipitation-measurement/",
        documentation: "https://gpm.nasa.gov/missions/GPM/gpm-core-observatory",
        reference_dimensions: [6.5,4.9,11.6], dimension_label: "Deployed", scale_mesh: None,
        dimension_source: "https://gpm.nasa.gov/sites/default/files/imce/GPM-Press-Kit_2014.pdf",
        facts: &facts::GPM, catalogue: catalogue::GPM, unassigned: &catalogue::GPM_OTHER,
    },
    Asset {
        name: "Suomi NPP", path: "assets/satellite-snpp.glb",
        credit: "NASA/Brian E. Kumanchik; NASA/Christian A. Lopez",
        source: "https://science.nasa.gov/3d-resources/suomi-national-polar-orbiting-partnership-suomi-npp/",
        documentation: "https://science.nasa.gov/mission/suomi-npp/",
        reference_dimensions: [1.3,1.3,4.2], dimension_label: "Body (approx.)", scale_mesh: Some(MeshScale::AssemblyAxis(&["foil_gold", "foil_silver"])),
        dimension_source: "https://www.eoportal.org/satellite-missions/suomi-npp#spacecraft",
        facts: &facts::SNPP, catalogue: catalogue::SNPP, unassigned: &catalogue::SNPP_OTHER,
    },
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub normal_uv: [f32; 2],
    pub roughness_uv: [f32; 2],
    pub color: [f32; 4],
    pub offset: [f32; 3],
    pub part: u32,
}

#[derive(Clone, Copy, Default, PartialEq)]
#[repr(u32)]
pub(crate) enum MaterialAppearance {
    #[default]
    Original,
    GoldReflector,
    ReflectorRim,
}

pub(crate) struct Material {
    pub name: String,
    pub color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub base_texture: Option<usize>,
    pub roughness_texture: Option<usize>,
    pub normal_texture: Option<usize>,
    pub normal_scale: f32,
    pub alpha_cutoff: f32,
    pub blend: bool,
    pub double_sided: bool,
    pub appearance: MaterialAppearance,
}

impl Material {
    pub fn is_blended(&self) -> bool {
        self.blend && self.appearance != MaterialAppearance::GoldReflector
    }

    pub fn rim_inner_radius(&self) -> f32 {
        if self.appearance == MaterialAppearance::ReflectorRim {
            0.87
        } else {
            0.0
        }
    }
}

#[derive(Clone)]
pub(crate) struct Part {
    pub name: String,
    pub component: usize,
    region: Option<usize>,
    inspectable: bool,
    pub start: u32,
    pub end: u32,
    pub material: usize,
    pub center: Vector3<f32>,
    pub offset: Vector3<f32>,
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

pub(crate) struct Component {
    pub info: &'static PartInfo,
    pub meshes: Vec<usize>,
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
    pub center: Vector3<f32>,
    pub visual_identification: bool,
}

impl Part {
    pub fn is_visible(&self, materials: &[Material]) -> bool {
        let material = &materials[self.material];
        self.inspectable && (!material.blend || material.color[3] > 0.0)
    }

    fn intersects_ray(
        &self,
        origin: Vector3<f32>,
        direction: Vector3<f32>,
        explosion: f32,
        nearest: f32,
    ) -> bool {
        let offset = self.offset * explosion;
        let (mut near, mut far) = (0.0_f32, nearest);
        for axis in 0..3 {
            let lo = self.min[axis] + offset[axis];
            let hi = self.max[axis] + offset[axis];
            if direction[axis].abs() < 1e-8 {
                if origin[axis] < lo || origin[axis] > hi {
                    return false;
                }
            } else {
                let a = (lo - origin[axis]) / direction[axis];
                let b = (hi - origin[axis]) / direction[axis];
                near = near.max(a.min(b));
                far = far.min(a.max(b));
                if near > far {
                    return false;
                }
            }
        }
        true
    }
}

pub(crate) struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub materials: Vec<Material>,
    pub images: Vec<image::RgbaImage>,
    pub parts: Vec<Part>,
    pub components: Vec<Component>,
    pub dimensions: Vector3<f32>,
    pub radius: f32,
    pub source_meshes: usize,
}

impl Model {
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > 64 * 1024 * 1024 {
            return Err("Model exceeds the 64 MiB import limit.".into());
        }
        let header = gltf::Gltf::from_slice_without_validation(bytes)
            .map_err(|e| format!("Invalid glTF model: {e}"))?;
        let expanded;
        let compressed_geometry = header
            .extensions_required()
            .any(|e| e == "KHR_draco_mesh_compression");
        let bytes = if compressed_geometry {
            let mut compressed = draco_gltf::parse(bytes, draco_gltf::ValidationProfile::Gltf20)
                .map_err(|e| format!("Cannot read compressed model: {e}"))?;
            compressed
                .decompress_in_place()
                .map_err(|e| format!("Cannot decompress model: {e}"))?;
            expanded = compressed
                .to_bytes(draco_gltf::OutputFormat::GlbV2)
                .map_err(|e| format!("Cannot materialize model: {e}"))?;
            expanded.as_slice()
        } else {
            bytes
        };
        // Draco's validated output retains unused compressed accessor declarations.
        // gltf-rs rejects those orphan declarations, although live accessors are materialized.
        let gltf = if compressed_geometry {
            gltf::Gltf::from_slice_without_validation(bytes)
        } else {
            gltf::Gltf::from_slice(bytes)
        }
        .map_err(|e| format!("Invalid glTF model: {e}"))?;
        if gltf
            .buffers()
            .any(|b| !matches!(b.source(), gltf::buffer::Source::Bin))
        {
            return Err("Use a self-contained GLB with embedded buffers and images.".into());
        }
        if gltf.skins().len() > 0 {
            return Err("Skinned models are not supported in the satellite inspector yet.".into());
        }
        let blob = gltf
            .blob
            .as_deref()
            .ok_or("The GLB has no embedded geometry.")?;
        if gltf.buffers().any(|b| b.length() > blob.len()) {
            return Err("The GLB geometry is truncated.".into());
        }
        let mut images = Vec::new();
        for source in gltf.images() {
            let encoded = match source.source() {
                gltf::image::Source::View { view, .. } => blob
                    .get(view.offset()..view.offset() + view.length())
                    .ok_or("Invalid embedded image range.")?
                    .to_vec(),
                gltf::image::Source::Uri { uri, .. } => {
                    let (header, body) = uri.split_once(',').ok_or(
                        "External image files are not supported; embed the textures in the GLB.",
                    )?;
                    if !header.starts_with("data:image/") || !header.ends_with(";base64") {
                        return Err("Unsupported embedded image encoding.".into());
                    }
                    base64::engine::general_purpose::STANDARD
                        .decode(body)
                        .map_err(|e| format!("Invalid image data: {e}"))?
                }
            };
            let mut reader = image::ImageReader::new(std::io::Cursor::new(encoded))
                .with_guessed_format()
                .map_err(|e| e.to_string())?;
            let mut limits = image::Limits::default();
            limits.max_image_width = Some(8192);
            limits.max_image_height = Some(8192);
            limits.max_alloc = Some(128 * 1024 * 1024);
            reader.limits(limits);
            let image = reader
                .decode()
                .map_err(|e| format!("Cannot decode model texture: {e}"))?
                .to_rgba8();
            images.push(image);
        }
        let mut materials: Vec<_> = gltf
            .materials()
            .map(|m| {
                let pbr = m.pbr_metallic_roughness();
                Material {
                    name: m.name().unwrap_or("Unnamed material").to_owned(),
                    color: pbr.base_color_factor(),
                    metallic: pbr.metallic_factor(),
                    roughness: pbr.roughness_factor().max(0.08),
                    base_texture: pbr
                        .base_color_texture()
                        .map(|t| t.texture().source().index()),
                    roughness_texture: pbr
                        .metallic_roughness_texture()
                        .map(|t| t.texture().source().index()),
                    normal_texture: m.normal_texture().map(|t| t.texture().source().index()),
                    normal_scale: m.normal_texture().map_or(0.0, |t| t.scale()),
                    alpha_cutoff: if m.alpha_mode() == gltf::material::AlphaMode::Mask {
                        m.alpha_cutoff().unwrap_or(0.5)
                    } else {
                        0.0
                    },
                    blend: m.alpha_mode() == gltf::material::AlphaMode::Blend,
                    double_sided: m.double_sided(),
                    appearance: MaterialAppearance::Original,
                }
            })
            .collect();
        materials.push(Material {
            name: "Default material".to_owned(),
            color: [1.0; 4],
            metallic: 0.0,
            roughness: 0.6,
            base_texture: None,
            roughness_texture: None,
            normal_texture: None,
            normal_scale: 0.0,
            alpha_cutoff: 0.0,
            blend: false,
            double_sided: false,
            appearance: MaterialAppearance::Original,
        });
        let mut model = Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            materials,
            images,
            parts: Vec::new(),
            components: Vec::new(),
            dimensions: Vector3::zeros(),
            radius: 1.0,
            source_meshes: 0,
        };
        let scene = gltf
            .default_scene()
            .or_else(|| gltf.scenes().next())
            .ok_or("The model has no scene.")?;
        for node in scene.nodes() {
            model.visit(node, Matrix4::identity(), blob)?;
        }
        if model.vertices.is_empty() {
            return Err("The model contains no triangle meshes.".into());
        }
        let mut min = Vector3::repeat(f32::INFINITY);
        let mut max = Vector3::repeat(f32::NEG_INFINITY);
        for v in &model.vertices {
            let p = Vector3::from(v.position);
            min = min.inf(&p);
            max = max.sup(&p);
        }
        model.dimensions = max - min;
        let center = (min + max) * 0.5;
        let scale = model
            .vertices
            .iter()
            .map(|v| (Vector3::from(v.position) - center).norm())
            .fold(0.0_f32, f32::max);
        if !scale.is_finite() || scale <= 1e-10 {
            return Err("The model has invalid or empty bounds.".into());
        }
        model.radius = scale;
        for part in &mut model.parts {
            part.center = (part.center - center) / scale;
            part.min = (part.min - center) / scale;
            part.max = (part.max - center) / scale;
            part.offset = part.center.try_normalize(1e-6).unwrap_or_else(Vector3::y) * 0.55;
            for &index in &model.indices[part.start as usize..part.end as usize] {
                // Vertices are unique to a primitive, but indices can refer to them repeatedly.
                model.vertices[index as usize].offset = part.offset.into();
            }
        }
        for vertex in &mut model.vertices {
            vertex.position = ((Vector3::from(vertex.position) - center) / scale).into();
        }
        model.source_meshes = model.parts.len();
        Ok(model)
    }

    fn partition_surfaces(&mut self, regions: &[catalogue::Region], protected: &[catalogue::Spec]) {
        if regions.is_empty() {
            return;
        }
        let original_indices = std::mem::take(&mut self.indices);
        let original_parts = std::mem::take(&mut self.parts);
        for part in original_parts {
            let triangles = &original_indices[part.start as usize..part.end as usize];
            let protected = protected.iter().any(|s| {
                s.meshes.contains(&part.name.as_str())
                    || s.materials
                        .contains(&self.materials[part.material].name.as_str())
            });
            let mut buckets: BTreeMap<Option<usize>, Vec<u32>> = BTreeMap::new();
            if protected {
                buckets.insert(None, triangles.to_vec());
            } else {
                let count = triangles.len() / 3;
                let mut parents: Vec<_> = (0..count).collect();
                let mut sizes = vec![1; count];
                let mut positions = HashMap::new();
                for (triangle, indices) in triangles.chunks_exact(3).enumerate() {
                    for &index in indices {
                        let key = self.vertices[index as usize]
                            .position
                            .map(|p| (p * 100_000.0).round() as i32);
                        if let Some(&other) = positions.get(&key) {
                            let mut a = surface_root(&mut parents, triangle);
                            let mut b = surface_root(&mut parents, other);
                            if a != b {
                                if sizes[a] < sizes[b] {
                                    std::mem::swap(&mut a, &mut b);
                                }
                                parents[b] = a;
                                sizes[a] += sizes[b];
                            }
                        } else {
                            positions.insert(key, triangle);
                        }
                    }
                }
                let mut islands: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
                for (triangle, indices) in triangles.chunks_exact(3).enumerate() {
                    islands
                        .entry(surface_root(&mut parents, triangle))
                        .or_default()
                        .extend_from_slice(indices);
                }
                for indices in islands.into_values() {
                    let (min, max) = mesh_bounds(&self.vertices, &indices);
                    let region = regions.iter().position(|r| {
                        (0..3).all(|axis| {
                            min[axis] >= r.min[axis] - 1e-5 && max[axis] <= r.max[axis] + 1e-5
                        })
                    });
                    buckets.entry(region).or_default().extend(indices);
                }
            }
            for (region, indices) in buckets {
                let (min, max) = mesh_bounds(&self.vertices, &indices);
                let mut split = part.clone();
                split.start = self.indices.len() as u32;
                self.indices.extend_from_slice(&indices);
                split.end = self.indices.len() as u32;
                split.region = region;
                split.min = min;
                split.max = max;
                split.center = (min + max) * 0.5;
                for index in indices {
                    self.vertices[index as usize].part = self.parts.len() as u32;
                }
                self.parts.push(split);
            }
        }
    }

    fn add_component(&mut self, info: &'static PartInfo, meshes: Vec<usize>) {
        let mut min = Vector3::repeat(f32::INFINITY);
        let mut max = Vector3::repeat(f32::NEG_INFINITY);
        for &index in &meshes {
            min = min.inf(&self.parts[index].min);
            max = max.sup(&self.parts[index].max);
        }
        let center = (min + max) * 0.5;
        let offset = center.try_normalize(1e-6).unwrap_or_else(Vector3::y) * 0.55;
        for &index in &meshes {
            self.parts[index].component = self.components.len();
            self.parts[index].offset = offset;
        }
        let visual_identification = meshes.iter().any(|&i| self.parts[i].region.is_some());
        self.components.push(Component {
            info,
            meshes,
            min,
            max,
            center,
            visual_identification,
        });
    }

    fn visit(
        &mut self,
        node: gltf::Node<'_>,
        parent: Matrix4<f32>,
        blob: &[u8],
    ) -> Result<(), String> {
        let local = Matrix4::from_columns(&node.transform().matrix().map(Vector4::from));
        let transform = parent * local;
        let normal_matrix = transform
            .fixed_view::<3, 3>(0, 0)
            .try_inverse()
            .map(|m| m.transpose())
            .unwrap_or_else(Matrix3::identity);
        if let Some(mesh) = node.mesh() {
            for (primitive_index, primitive) in mesh.primitives().enumerate() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let reader = primitive.reader(|_| Some(blob));
                let positions: Vec<_> = reader
                    .read_positions()
                    .ok_or("A mesh has no positions.")?
                    .collect();
                if positions.iter().flatten().any(|x| !x.is_finite()) {
                    return Err("The model contains non-finite vertex positions.".into());
                }
                let normals: Option<Vec<_>> = reader.read_normals().map(Iterator::collect);
                let texture_info = primitive
                    .material()
                    .pbr_metallic_roughness()
                    .base_color_texture();
                let texture_transform = texture_info.as_ref().and_then(|t| t.texture_transform());
                let tex_coord = texture_transform
                    .as_ref()
                    .and_then(|t| t.tex_coord())
                    .unwrap_or_else(|| texture_info.as_ref().map_or(0, |t| t.tex_coord()));
                let mut texture_coords: Vec<[f32; 2]> = reader
                    .read_tex_coords(tex_coord)
                    .map(|uv| uv.into_f32().collect())
                    .unwrap_or_default();
                if let Some(transform) = texture_transform {
                    let [sx, sy] = transform.scale();
                    let [ox, oy] = transform.offset();
                    let (s, c) = transform.rotation().sin_cos();
                    for uv in &mut texture_coords {
                        *uv = [
                            ox + c * sx * uv[0] - s * sy * uv[1],
                            oy + s * sx * uv[0] + c * sy * uv[1],
                        ];
                    }
                }
                let normal_info = primitive.material().normal_texture();
                let normal_transform = normal_info
                    .as_ref()
                    .and_then(|t| t.extension_value("KHR_texture_transform"))
                    .map(|value| serde_json::from_value::<UvTransform>(value.clone()))
                    .transpose()
                    .map_err(|e| format!("Invalid normal texture transform: {e}"))?
                    .unwrap_or_default();
                let mut normal_uv: Vec<[f32; 2]> = reader
                    .read_tex_coords(
                        normal_transform
                            .tex_coord
                            .unwrap_or_else(|| normal_info.as_ref().map_or(0, |t| t.tex_coord())),
                    )
                    .map(|uv| uv.into_f32().collect())
                    .unwrap_or_default();
                normal_transform.apply(&mut normal_uv);
                let roughness_info = primitive
                    .material()
                    .pbr_metallic_roughness()
                    .metallic_roughness_texture();
                let roughness_transform =
                    roughness_info.as_ref().and_then(|t| t.texture_transform());
                let mut roughness_uv: Vec<[f32; 2]> = reader
                    .read_tex_coords(
                        roughness_transform
                            .as_ref()
                            .and_then(|t| t.tex_coord())
                            .unwrap_or_else(|| {
                                roughness_info.as_ref().map_or(0, |t| t.tex_coord())
                            }),
                    )
                    .map(|uv| uv.into_f32().collect())
                    .unwrap_or_default();
                if let Some(t) = roughness_transform {
                    UvTransform {
                        offset: t.offset(),
                        scale: t.scale(),
                        rotation: t.rotation(),
                        tex_coord: t.tex_coord(),
                    }
                    .apply(&mut roughness_uv);
                }
                let colors: Vec<_> = reader
                    .read_colors(0)
                    .map(|c| c.into_rgba_f32().collect())
                    .unwrap_or_default();
                let mut indices: Vec<u32> = reader
                    .read_indices()
                    .map(|i| i.into_u32().collect())
                    .unwrap_or_else(|| (0..positions.len() as u32).collect());
                if indices.len() % 3 != 0 || indices.iter().any(|&i| i as usize >= positions.len())
                {
                    return Err("The model has invalid triangle indices.".into());
                }
                if transform.fixed_view::<3, 3>(0, 0).determinant() < 0.0 {
                    for triangle in indices.chunks_exact_mut(3) {
                        triangle.swap(1, 2);
                    }
                }
                if self.vertices.len() + positions.len() > 2_000_000 {
                    return Err("The model exceeds the two-million-vertex limit.".into());
                }
                let part = self.parts.len() as u32;
                let base = self.vertices.len() as u32;
                let start = self.indices.len() as u32;
                let mut min = Vector3::repeat(f32::INFINITY);
                let mut max = Vector3::repeat(f32::NEG_INFINITY);
                for (i, position) in positions.iter().enumerate() {
                    let p = transform * Vector4::new(position[0], position[1], position[2], 1.0);
                    let p = p.xyz();
                    min = min.inf(&p);
                    max = max.sup(&p);
                    let normal = normals
                        .as_ref()
                        .and_then(|n| n.get(i))
                        .map(|&n| normal_matrix * Vector3::from(n))
                        .unwrap_or_default();
                    self.vertices.push(Vertex {
                        position: p.into(),
                        normal: if normals.is_some() {
                            normal.try_normalize(1e-8).unwrap_or_else(Vector3::y).into()
                        } else {
                            [0.0; 3]
                        },
                        uv: texture_coords.get(i).copied().unwrap_or([0.0; 2]),
                        normal_uv: normal_uv.get(i).copied().unwrap_or([0.0; 2]),
                        roughness_uv: roughness_uv.get(i).copied().unwrap_or([0.0; 2]),
                        color: colors.get(i).copied().unwrap_or([1.0; 4]),
                        offset: [0.0; 3],
                        part,
                    });
                }
                if normals.is_none() {
                    for triangle in indices.chunks_exact(3) {
                        let p = [0, 1, 2].map(|i| {
                            Vector3::from(self.vertices[(base + triangle[i]) as usize].position)
                        });
                        let normal = (p[1] - p[0]).cross(&(p[2] - p[0]));
                        for &i in triangle {
                            let vertex = &mut self.vertices[(base + i) as usize];
                            vertex.normal = (Vector3::from(vertex.normal) + normal).into();
                        }
                    }
                    for v in &mut self.vertices[base as usize..] {
                        v.normal = Vector3::from(v.normal)
                            .try_normalize(1e-8)
                            .unwrap_or_else(Vector3::y)
                            .into();
                    }
                }
                self.indices.extend(indices.into_iter().map(|i| base + i));
                let name = node
                    .name()
                    .or(mesh.name())
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("Mesh {}", mesh.index() + 1));
                self.parts.push(Part {
                    inspectable: true,
                    region: None,
                    component: part as usize,
                    name: if primitive_index == 0 {
                        name
                    } else {
                        format!("{name} / {}", primitive_index + 1)
                    },
                    start,
                    end: self.indices.len() as u32,
                    material: primitive
                        .material()
                        .index()
                        .unwrap_or(self.materials.len() - 1),
                    center: (min + max) * 0.5,
                    offset: Vector3::zeros(),
                    min,
                    max,
                });
            }
        }
        for child in node.children() {
            self.visit(child, transform, blob)?;
        }
        Ok(())
    }

    pub fn pick(
        &self,
        origin: Vector3<f32>,
        direction: Vector3<f32>,
        explosion: f32,
        isolate: Option<usize>,
    ) -> Option<usize> {
        let mut nearest = f32::INFINITY;
        let mut picked = None;
        for part in &self.parts {
            if !part.is_visible(&self.materials) {
                continue;
            }
            if isolate.is_some_and(|i| i != part.component) {
                continue;
            }
            if !part.intersects_ray(origin, direction, explosion, nearest) {
                continue;
            }
            let offset = part.offset * explosion;
            for triangle in self.indices[part.start as usize..part.end as usize].chunks_exact(3) {
                let p = [0, 1, 2]
                    .map(|i| Vector3::from(self.vertices[triangle[i] as usize].position) + offset);
                let e1 = p[1] - p[0];
                let e2 = p[2] - p[0];
                let cross = direction.cross(&e2);
                let determinant = e1.dot(&cross);
                if determinant.abs() < 1e-8
                    || (!self.materials[part.material].double_sided && determinant < 0.0)
                {
                    continue;
                }
                let inv = determinant.recip();
                let t = origin - p[0];
                let u = t.dot(&cross) * inv;
                if !(0.0..=1.0).contains(&u) {
                    continue;
                }
                let q = t.cross(&e1);
                let v = direction.dot(&q) * inv;
                if v < 0.0 || u + v > 1.0 {
                    continue;
                }
                let radius = self.materials[part.material].rim_inner_radius();
                if radius > 0.0 {
                    let uv = [0, 1, 2]
                        .map(|i| nalgebra::Vector2::from(self.vertices[triangle[i] as usize].uv));
                    let uv = uv[0] * (1.0 - u - v) + uv[1] * u + uv[2] * v;
                    if ((uv - nalgebra::Vector2::repeat(0.5)) * 2.0).norm_squared()
                        < radius * radius
                    {
                        continue;
                    }
                }
                let distance = e2.dot(&q) * inv;
                if distance > 0.0 && distance < nearest {
                    nearest = distance;
                    picked = Some(part.component);
                }
            }
        }
        picked
    }
}

fn surface_root(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}

fn mesh_bounds(vertices: &[Vertex], indices: &[u32]) -> (Vector3<f32>, Vector3<f32>) {
    let mut min = Vector3::repeat(f32::INFINITY);
    let mut max = Vector3::repeat(f32::NEG_INFINITY);
    for &index in indices {
        let p = Vector3::from(vertices[index as usize].position);
        min = min.inf(&p);
        max = max.sup(&p);
    }
    (min, max)
}

#[derive(serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct UvTransform {
    offset: [f32; 2],
    scale: [f32; 2],
    rotation: f32,
    tex_coord: Option<u32>,
}
impl Default for UvTransform {
    fn default() -> Self {
        Self {
            offset: [0.0; 2],
            scale: [1.0; 2],
            rotation: 0.0,
            tex_coord: None,
        }
    }
}
impl UvTransform {
    fn apply(&self, coordinates: &mut [[f32; 2]]) {
        let [sx, sy] = self.scale;
        let [ox, oy] = self.offset;
        let (s, c) = self.rotation.sin_cos();
        for uv in coordinates {
            *uv = [
                ox + c * sx * uv[0] - s * sy * uv[1],
                oy + s * sx * uv[0] + c * sy * uv[1],
            ];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smap_simplification_changes_only_display_materials_not_the_downloaded_mesh() {
        let bytes = std::fs::read(crate::texture::asset_path(ASSETS[4].path)).unwrap();
        let source = Model::load(&bytes).unwrap();
        let display = ASSETS[4].load(&bytes).unwrap();
        for (original, shown) in source.materials.iter().zip(&display.materials) {
            assert!(original.appearance == MaterialAppearance::Original);
            assert_eq!(shown.double_sided, original.double_sided);
            assert_eq!(shown.color, original.color);
            assert_eq!(shown.base_texture, original.base_texture);
            match shown.name.as_str() {
                "antenna_a" => {
                    assert!(shown.appearance == MaterialAppearance::GoldReflector);
                    assert!(
                        !shown.is_blended(),
                        "the simplified gold surface must occlude rear trusses"
                    );
                }
                "antenna_support_structure" => {
                    assert!(shown.appearance == MaterialAppearance::ReflectorRim)
                }
                _ => assert!(shown.appearance == MaterialAppearance::Original),
            }
        }
        assert_eq!(display.vertices.len(), source.vertices.len());
        assert!(display
            .vertices
            .iter()
            .zip(&source.vertices)
            .all(|(a, b)| a.position == b.position && a.uv == b.uv));
    }

    #[test]
    fn picking_respects_single_and_double_sided_surfaces() {
        let mut model =
            Model::load(&std::fs::read(crate::texture::asset_path(ASSETS[0].path)).unwrap())
                .unwrap();
        let vertex = model.vertices[0];
        model.vertices = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
            .map(|position| Vertex { position, ..vertex })
            .to_vec();
        model.indices = vec![0, 1, 2];
        model.parts.truncate(1);
        let part = &mut model.parts[0];
        part.start = 0;
        part.end = 3;
        part.component = 0;
        part.material = 0;
        part.inspectable = true;
        part.min = Vector3::new(-1.0, -1.0, 0.0);
        part.max = Vector3::new(1.0, 1.0, 0.0);
        model.materials[0].double_sided = false;
        assert_eq!(
            model.pick(Vector3::new(0.0, 0.0, 2.0), -Vector3::z(), 0.0, None),
            Some(0)
        );
        assert_eq!(
            model.pick(Vector3::new(0.0, 0.0, -2.0), Vector3::z(), 0.0, None),
            None
        );
        model.materials[0].double_sided = true;
        assert_eq!(
            model.pick(Vector3::new(0.0, 0.0, -2.0), Vector3::z(), 0.0, None),
            Some(0)
        );
        model.materials[0].appearance = MaterialAppearance::ReflectorRim;
        for vertex in &mut model.vertices {
            vertex.uv = [0.5, 0.5];
        }
        assert_eq!(
            model.pick(Vector3::new(0.0, 0.0, 2.0), -Vector3::z(), 0.0, None),
            None,
            "the hidden interior web must not remain clickable"
        );
        for vertex in &mut model.vertices {
            vertex.uv = [1.0, 0.5];
        }
        assert_eq!(
            model.pick(Vector3::new(0.0, 0.0, 2.0), -Vector3::z(), 0.0, None),
            Some(0),
            "the visible perimeter truss must remain clickable"
        );
    }

    #[test]
    fn reviewed_components_keep_all_original_geometry_and_document_uncertainty() {
        for asset in &ASSETS {
            let bytes = std::fs::read(crate::texture::asset_path(asset.path)).unwrap();
            let original = Model::load(&bytes).unwrap();
            let model = asset.load(&bytes).unwrap();
            let triangles = |m: &Model| {
                let mut values: Vec<_> = m
                    .parts
                    .iter()
                    .flat_map(|part| {
                        m.indices[part.start as usize..part.end as usize]
                            .chunks_exact(3)
                            .map(|t| (part.material, [t[0], t[1], t[2]]))
                    })
                    .collect();
                values.sort_unstable();
                values
            };
            assert_eq!(
                triangles(&model),
                triangles(&original),
                "annotation must preserve every original triangle, winding and material"
            );
            assert_eq!(model.source_meshes, original.parts.len());
            assert!(
                (asset.reference_span(&model) - asset.reference_span(&original)).abs() < 1e-5,
                "component grouping must preserve the full source scale reference"
            );
            assert_eq!(model.vertices.len(), original.vertices.len());
            let mut membership = vec![0; model.parts.len()];
            assert!(!asset.facts.rows.is_empty());
            assert!(!asset.facts.sources.is_empty());
            for (index, component) in model.components.iter().enumerate() {
                assert!(!component.info.description.is_empty());
                assert!(!component.info.review.is_empty());
                assert!(component.info.source.starts_with("https://"));
                for &mesh in &component.meshes {
                    membership[mesh] += 1;
                    let part = &model.parts[mesh];
                    assert_eq!(part.component, index);
                    for &vertex in &model.indices[part.start as usize..part.end as usize] {
                        let reviewed = &model.vertices[vertex as usize];
                        assert_eq!(reviewed.part as usize, index);
                        assert_eq!(
                            reviewed.position,
                            original.vertices[vertex as usize].position
                        );
                        assert_eq!(reviewed.offset, <[f32; 3]>::from(part.offset));
                    }
                }
            }
            assert!(membership
                .iter()
                .zip(&model.parts)
                .all(|(&n, p)| n == usize::from(p.is_visible(&model.materials))));
            for spec in asset.catalogue {
                for &name in spec.meshes {
                    assert!(
                        model.parts.iter().any(|p| name
                            .strip_suffix('*')
                            .map_or(p.name == name, |prefix| p.name.starts_with(prefix))),
                        "missing reviewed mesh {name}"
                    );
                }
                for &material in spec.materials {
                    assert!(
                        model.materials.iter().any(|m| m.name == material),
                        "missing reviewed material {material}"
                    );
                }
            }
            let physical_span = asset.reference_span(&model) * asset.metres_per_unit(&model);
            assert!(
                (physical_span
                    - asset
                        .reference_dimensions
                        .into_iter()
                        .fold(0.0_f32, f32::max))
                .abs()
                    < 1e-4
            );
        }
        let model = ASSETS[0]
            .load(&std::fs::read(crate::texture::asset_path(ASSETS[0].path)).unwrap())
            .unwrap();
        assert_eq!(model.components.len(), 14);
        let mixed = model.parts.iter().find(|p| p.name == "AMR / 3").unwrap();
        assert_eq!(
            model.components[mixed.component].info.title,
            "Other exterior details"
        );
        let bus = model
            .components
            .iter()
            .find(|c| c.info.title == "Spacecraft bus")
            .unwrap();
        assert_eq!(bus.meshes.len(), 3);
        let array = model
            .components
            .iter()
            .find(|c| c.info.title == "Solar array")
            .unwrap();
        assert_eq!(array.meshes.len(), 6);
    }

    #[test]
    fn picking_bounds_handle_parallel_rays_and_exploded_positions() {
        let part = Part {
            name: String::new(),
            component: 0,
            region: None,
            inspectable: true,
            start: 0,
            end: 0,
            material: 0,
            center: Vector3::zeros(),
            offset: Vector3::new(3.0, 0.0, 0.0),
            min: Vector3::repeat(-1.0),
            max: Vector3::repeat(1.0),
        };
        let origin = Vector3::new(0.0, 0.0, 3.0);
        assert!(part.intersects_ray(origin, -Vector3::z(), 0.0, f32::INFINITY));
        assert!(!part.intersects_ray(origin, Vector3::z(), 0.0, f32::INFINITY));
        assert!(!part.intersects_ray(origin, -Vector3::z(), 0.0, 1.0));
        assert!(!part.intersects_ray(origin, -Vector3::z(), 1.0, f32::INFINITY));
        assert!(part.intersects_ray(
            origin + Vector3::new(3.0, 0.0, 0.0),
            -Vector3::z(),
            1.0,
            f32::INFINITY
        ));
    }

    #[test]
    fn downloaded_nasa_models_load_with_original_materials_and_meshes() {
        for asset in &ASSETS {
            let bytes = std::fs::read(crate::texture::asset_path(asset.path)).unwrap();
            let model = Model::load(&bytes).unwrap();
            eprintln!("{}: {} vertices, {} triangles, {} parts, {} materials, {} textures; dimensions {:?}", asset.name, model.vertices.len(), model.indices.len()/3, model.parts.len(), model.materials.len(), model.images.len(), model.dimensions);
            assert!(model.parts.len() > 1);
            assert!(model.indices.len() > 300);
            assert_eq!(std::mem::size_of::<Vertex>(), 80);
            assert!(model
                .vertices
                .iter()
                .all(|v| Vector3::from(v.position).norm() <= 1.00001));
            assert!(model
                .vertices
                .iter()
                .all(|v| v.normal.iter().all(|n| n.is_finite())));
        }
    }

    #[test]
    fn additional_earth_observers_have_reviewed_parts_and_physical_scales() {
        for asset in &ASSETS[9..] {
            let model = asset
                .load(&std::fs::read(crate::texture::asset_path(asset.path)).unwrap())
                .unwrap();
            assert!(
                model.components.len() >= 4,
                "{} has no useful part assignments",
                asset.name
            );
            let metres_per_unit = asset.metres_per_unit(&model);
            assert!(metres_per_unit.is_finite() && metres_per_unit > 0.0);
            if asset.name == "GPM Core Observatory" {
                for (title, min_y, max_y) in [
                    ("High-gain communications antenna", 0.335, 0.51),
                    ("GMI microwave imager", -0.03, 0.335),
                    ("Dual-frequency precipitation radar", -0.31, 0.09),
                ] {
                    let component = model
                        .components
                        .iter()
                        .find(|c| c.info.title == title)
                        .unwrap();
                    assert!(component.min.y >= min_y && component.max.y <= max_y);
                }
                assert!(
                    model.parts.len() > model.source_meshes,
                    "mixed source finishes need connected-surface assignments"
                );
            }
            if asset.name == "Suomi NPP" {
                let body_height = asset.reference_span(&model) * metres_per_unit;
                let deployed_span = model.dimensions.max() / model.radius * metres_per_unit;
                assert!((body_height - 4.2).abs() < 1e-4);
                assert!(
                    deployed_span > body_height * 2.0,
                    "the solar wing must not be scaled down to the published body height"
                );
                assert!(model
                    .components
                    .iter()
                    .any(|c| c.info.title == "Solar-array back and frame"));
            }
            if asset.name == "Aura" {
                assert!(model
                    .components
                    .iter()
                    .any(|c| c.info.title == "Microwave Limb Sounder (MLS)"));
                assert_eq!(
                    model
                        .components
                        .iter()
                        .filter(|c| c.info.title.contains("retired"))
                        .count(),
                    2
                );
            }
        }
    }

    #[test]
    fn invalid_models_return_an_error() {
        assert!(Model::load(b"not a GLB").is_err());
        assert!(Model::load(
            br#"{"asset":{"version":"2.0"},"buffers":[{"uri":"external.bin","byteLength":12}]}"#
        )
        .is_err());
    }

    #[test]
    fn added_models_have_reviewed_components_and_hidden_authoring_helpers() {
        for (index, titles) in [
            (
                6,
                &[
                    "Advanced Baseline Imager (ABI)",
                    "Geostationary Lightning Mapper (GLM)",
                    "Magnetometer sensor heads",
                    "Solar observatories (SUVI / EXIS)",
                ][..],
            ),
            (
                7,
                &[
                    "Solar wing A",
                    "Solar wing B",
                    "KaRIn antenna A",
                    "KaRIn antenna B",
                    "Nadir radar altimeter",
                    "Microwave radiometer",
                    "X-band downlink antenna",
                ][..],
            ),
            (
                8,
                &[
                    "Microwave ranging horn",
                    "Zenith navigation antenna",
                    "Aft radio-occultation antenna",
                    "Star trackers and protective baffles",
                    "Laser retroreflector",
                ][..],
            ),
        ] {
            let asset = &ASSETS[index];
            let model = asset
                .load(&std::fs::read(crate::texture::asset_path(asset.path)).unwrap())
                .unwrap();
            for title in titles {
                assert!(
                    model
                        .components
                        .iter()
                        .any(|c| c.info.title == *title && !c.meshes.is_empty()),
                    "{} is missing {title}",
                    asset.name
                );
            }
            if index != 6 {
                let helper = model.parts.iter().position(|p| p.name == "_root").unwrap();
                assert!(!model.parts[helper].is_visible(&model.materials));
                assert!(model.components.iter().all(|c| !c.meshes.contains(&helper)));
            }
        }
    }

    #[test]
    fn aqua_has_separate_educational_heads_and_smap_helpers_are_not_parts() {
        let aqua = ASSETS[1]
            .load(&std::fs::read(crate::texture::asset_path(ASSETS[1].path)).unwrap())
            .unwrap();
        for title in [
            "X-band communications dish",
            "CERES scan head 1",
            "CERES scan head 2",
            "MODIS housing and Earth shield",
            "HSB-region sounding head",
            "Microwave sounding head",
        ] {
            let component = aqua
                .components
                .iter()
                .find(|c| c.info.title == title)
                .unwrap_or_else(|| panic!("missing educational component {title}"));
            assert!(component.visual_identification);
            assert!(!component.meshes.is_empty());
        }
        assert!(aqua.parts.len() > aqua.source_meshes);
        let smap = ASSETS[4]
            .load(&std::fs::read(crate::texture::asset_path(ASSETS[4].path)).unwrap())
            .unwrap();
        let helper = smap.parts.iter().position(|p| p.name == "_root").unwrap();
        assert!(!smap.parts[helper].is_visible(&smap.materials));
        assert!(smap.components.iter().all(|c| !c.meshes.contains(&helper)));
    }

    #[test]
    #[ignore = "prints connected-surface bounds for reviewing additional models"]
    fn inspect_additional_surface_bounds() {
        for asset in ASSETS.iter().skip(7) {
            let name = asset.name;
            let model =
                Model::load(&std::fs::read(crate::texture::asset_path(asset.path)).unwrap())
                    .unwrap();
            for part in &model.parts {
                if part.name == "_root" {
                    continue;
                }
                let triangles = &model.indices[part.start as usize..part.end as usize];
                let mut parents: Vec<_> = (0..triangles.len() / 3).collect();
                let mut positions = HashMap::new();
                for (triangle, indices) in triangles.chunks_exact(3).enumerate() {
                    for &index in indices {
                        let key = model.vertices[index as usize]
                            .position
                            .map(|p| (p * 100_000.0).round() as i32);
                        if let Some(&other) = positions.get(&key) {
                            let a = surface_root(&mut parents, triangle);
                            let b = surface_root(&mut parents, other);
                            parents[b] = a;
                        } else {
                            positions.insert(key, triangle);
                        }
                    }
                }
                let mut islands: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
                for (triangle, indices) in triangles.chunks_exact(3).enumerate() {
                    islands
                        .entry(surface_root(&mut parents, triangle))
                        .or_default()
                        .extend_from_slice(indices);
                }
                eprintln!("{name} {}: {} surfaces", part.name, islands.len());
                for (index, indices) in islands.values().enumerate() {
                    let (min, max) = mesh_bounds(&model.vertices, indices);
                    if indices.len() >= 24 || (max - min).norm() > 0.2 {
                        eprintln!(
                            "{index}: {} triangles, min {:?}, max {:?}",
                            indices.len() / 3,
                            min,
                            max
                        );
                    }
                }
            }
        }
    }
}
