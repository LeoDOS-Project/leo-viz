use egui_wgpu::wgpu;
use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use nalgebra::Matrix3;
use std::collections::HashMap;

use crate::celestial::{CelestialBody, Skin, TextureResolution};
use crate::texture::{EarthTexture, RingTexture};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlanetUniforms {
    pub inv_rot_0: [f32; 4],
    pub inv_rot_1: [f32; 4],
    pub inv_rot_2: [f32; 4],
    pub star_rot_0: [f32; 4],
    pub star_rot_1: [f32; 4],
    pub star_rot_2: [f32; 4],
    pub sun_dir_flat: [f32; 4],
    pub bg_aspect: [f32; 4],
    pub detail_bounds: [f32; 4],
    pub uv_etc: [f32; 4],
    pub flags_a: [f32; 4],
    pub flags_b: [f32; 4],
    pub flags_c: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SunUniforms {
    pub uv_aspect_ps: [f32; 4],
    pub sun_cam_int: [f32; 4],
    pub zoom_pad: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HeatmapUniforms {
    pub inv_rot_0: [f32; 4],
    pub inv_rot_1: [f32; 4],
    pub inv_rot_2: [f32; 4],
    pub mag_aspect: [f32; 4],
    pub dipole_scale: [f32; 4],
    pub uv_mode_smooth: [f32; 4],
    pub kp_pr_sr_sp: [f32; 4],
    pub se_pad: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MapUniforms {
    pub bounds: [f32; 4],
    pub peirce: [f32; 4],
    pub proj_invscale: [f32; 4],
}

pub struct RttRequest {
    pub key: (CelestialBody, Skin, TextureResolution),
    pub inv_rotation: Matrix3<f64>,
    pub flattening: f64,
    pub size: usize,
    pub skip_rings: bool,
}

pub struct GpuResources {
    pub planet_pipeline: wgpu::RenderPipeline,
    pub rtt_pipeline: wgpu::RenderPipeline,
    pub sun_pipeline: wgpu::RenderPipeline,
    pub heatmap_pipeline: wgpu::RenderPipeline,
    pub map_pipeline: wgpu::RenderPipeline,

    pub planet_bgl: wgpu::BindGroupLayout,
    pub _sun_bgl: wgpu::BindGroupLayout,
    pub heatmap_bgl: wgpu::BindGroupLayout,
    pub map_bgl: wgpu::BindGroupLayout,

    pub sampler_repeat: wgpu::Sampler,
    pub sampler_clamp: wgpu::Sampler,

    pub textures: HashMap<(CelestialBody, Skin, TextureResolution), wgpu::TextureView>,
    pub cloud_textures: HashMap<TextureResolution, wgpu::TextureView>,
    pub night_texture: Option<wgpu::TextureView>,
    pub star_texture: Option<wgpu::TextureView>,
    pub milky_way_texture: Option<wgpu::TextureView>,
    pub ring_textures: HashMap<CelestialBody, wgpu::TextureView>,
    pub heatmap_palette_view: wgpu::TextureView,
    pub heatmap_data_view: Option<wgpu::TextureView>,
    pub heatmap_data_key: u64,
    pub map_texture_view: Option<wgpu::TextureView>,
    pub map_texture_key: Option<(CelestialBody, Skin, TextureResolution)>,
    pub detail_view: Option<wgpu::TextureView>,

    pub igrf_coeffs_buf: wgpu::Buffer,
    pub dummy_view: wgpu::TextureView,
    pub _dummy_view_rgba: wgpu::TextureView,

    pub _target_format: wgpu::TextureFormat,

    pub sun_ub: wgpu::Buffer,
    pub planet_bg_queue: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    pub planet_bg_paint_idx: std::sync::atomic::AtomicUsize,
    pub heatmap_bg_queue: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    pub heatmap_bg_paint_idx: std::sync::atomic::AtomicUsize,
    pub map_bg_queue: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    pub map_bg_paint_idx: std::sync::atomic::AtomicUsize,
    pub sun_bg: Option<wgpu::BindGroup>,
    pub texture_gen: u64,

    pub rad_trace_pipeline: wgpu::ComputePipeline,
    pub rad_blur_pipeline: wgpu::ComputePipeline,
    pub rad_reduce_pipeline: wgpu::ComputePipeline,
    pub rad_finalize_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    pub rad_compute_bgl: wgpu::BindGroupLayout,
    pub rad_bg_ab: wgpu::BindGroup,
    pub rad_bg_ba: wgpu::BindGroup,
    pub rad_params_buf: wgpu::Buffer,
    #[allow(dead_code)]
    pub rad_aep8_buf: wgpu::Buffer,
    #[allow(dead_code)]
    pub rad_grid_a: wgpu::Buffer,
    #[allow(dead_code)]
    pub rad_grid_b: wgpu::Buffer,
    pub rad_max_buf: wgpu::Buffer,
    pub rad_output_tex: wgpu::Texture,
    #[allow(dead_code)]
    pub rad_output_view: wgpu::TextureView,
    pub rad_computed_params: Option<f32>,
}

fn create_dummy_texture(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dummy"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let data = match format {
        wgpu::TextureFormat::Rgba8UnormSrgb => vec![0u8, 0, 0, 255],
        _ => vec![0u8, 0, 0, 255],
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &data,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: None },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn upload_rgb_texture(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, rgb: &[u8]) -> wgpu::TextureView {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for chunk in rgb.chunks(3) {
        rgba.push(chunk[0]);
        rgba.push(chunk[1]);
        rgba.push(chunk[2]);
        rgba.push(255);
    }
    upload_rgba_texture(device, queue, width, height, &rgba)
}

fn downscale_rgba(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dw * dh * 4) as usize];
    let x_ratio = sw as f64 / dw as f64;
    let y_ratio = sh as f64 / dh as f64;
    for dy in 0..dh {
        for dx in 0..dw {
            let sx = ((dx as f64 + 0.5) * x_ratio) as u32;
            let sy = ((dy as f64 + 0.5) * y_ratio) as u32;
            let si = (sy * sw + sx) as usize * 4;
            let di = (dy * dw + dx) as usize * 4;
            dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }
    dst
}

fn upload_rgba_texture(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, rgba: &[u8]) -> wgpu::TextureView {
    let max_dim = device.limits().max_texture_dimension_2d;
    let (uw, uh, data);
    if width > max_dim || height > max_dim {
        let scale = (max_dim as f64 / width.max(height) as f64).min(1.0);
        uw = ((width as f64 * scale) as u32).max(1);
        uh = ((height as f64 * scale) as u32).max(1);
        data = downscale_rgba(rgba, width, height, uw, uh);
    } else {
        uw = width;
        uh = height;
        data = rgba.to_vec();
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: uw, height: uh, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &data,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(uw * 4), rows_per_image: None },
        wgpu::Extent3d { width: uw, height: uh, depth_or_array_layers: 1 },
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

impl GpuResources {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let sampler_repeat = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler_repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sampler_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler_clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let planet_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_sampler(1),
                bgl_sampler(2),
                bgl_texture(3),
                bgl_texture(4),
                bgl_texture(5),
                bgl_texture(6),
                bgl_texture(7),
                bgl_texture(8),
            ],
        });
        let sun_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sun_bgl"),
            entries: &[bgl_uniform(0)],
        });
        let heatmap_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("heatmap_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_sampler(1),
                bgl_texture(2),
                bgl_texture(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let map_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("map_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_sampler(1),
                bgl_texture(2),
            ],
        });

        let planet_pipeline = create_pipeline(device, format, &planet_bgl, PLANET_WGSL, "planet",
            Some(wgpu::BlendState::ALPHA_BLENDING));
        let rtt_pipeline = create_pipeline(device, wgpu::TextureFormat::Rgba8Unorm, &planet_bgl, PLANET_WGSL, "rtt",
            None);
        let sun_pipeline = create_pipeline(device, format, &sun_bgl, SUN_WGSL, "sun",
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }));
        let heatmap_pipeline = create_pipeline(device, format, &heatmap_bgl, HEATMAP_WGSL, "heatmap",
            Some(wgpu::BlendState::ALPHA_BLENDING));
        let map_pipeline = create_pipeline(device, format, &map_bgl, MAP_WGSL, "map",
            Some(wgpu::BlendState::ALPHA_BLENDING));

        let dummy_view = create_dummy_texture(device, queue, wgpu::TextureFormat::Rgba8Unorm);
        let dummy_view_rgba = create_dummy_texture(device, queue, wgpu::TextureFormat::Rgba8Unorm);

        let palette = &crate::config::GEOMAGNETIC_PALETTE;
        let mut palette_rgba = Vec::with_capacity(palette.len() * 4);
        for &[r, g, b] in palette.iter() {
            palette_rgba.push(r);
            palette_rgba.push(g);
            palette_rgba.push(b);
            palette_rgba.push(255);
        }
        let heatmap_palette_view = upload_rgba_texture(device, queue, palette.len() as u32, 1, &palette_rgba);

        let igrf_coeffs: Vec<f32> = crate::igrf::IGRF_GC.iter()
            .chain(crate::igrf::IGRF_HC.iter())
            .copied().collect();
        let igrf_coeffs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("igrf_coeffs"),
            size: (igrf_coeffs.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&igrf_coeffs_buf, 0, bytemuck::cast_slice(&igrf_coeffs));

        let make_ub = |label, size| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sun_ub = make_ub("sun_ub", std::mem::size_of::<SunUniforms>() as u64);

        let sun_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sun_bg"),
            layout: &sun_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: sun_ub.as_entire_binding() }],
        }));

        let rad_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rad_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        use crate::aep8::{AP8MAX_DESCR, AP8MAX_MAP, AE8MAX_DESCR, AE8MAX_MAP};
        let mut aep8_data: Vec<i32> = Vec::with_capacity(16 + AP8MAX_MAP.len() + AE8MAX_MAP.len());
        aep8_data.extend_from_slice(&AP8MAX_DESCR);
        aep8_data.extend_from_slice(&AE8MAX_DESCR);
        aep8_data.extend_from_slice(&AP8MAX_MAP);
        aep8_data.extend_from_slice(&AE8MAX_MAP);
        let rad_aep8_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rad_aep8"),
            size: (aep8_data.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&rad_aep8_buf, 0, bytemuck::cast_slice(&aep8_data));

        let grid_size = (91 * 181 * 2 * 4) as u64;
        let rad_grid_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rad_grid_a"), size: grid_size,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let rad_grid_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rad_grid_b"), size: grid_size,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let rad_max_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rad_max"), size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let rad_output_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rad_output"),
            size: wgpu::Extent3d { width: 181, height: 91, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let rad_output_view = rad_output_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let rad_compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rad_compute_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let rad_compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rad_compute"),
            source: wgpu::ShaderSource::Wgsl(RAD_COMPUTE_WGSL.into()),
        });

        let rad_trace_pipeline = create_compute_pipeline(device, &rad_compute_bgl, &rad_compute_shader, "trace_main", "rad_trace");
        let rad_blur_pipeline = create_compute_pipeline(device, &rad_compute_bgl, &rad_compute_shader, "blur_main", "rad_blur");
        let rad_reduce_pipeline = create_compute_pipeline(device, &rad_compute_bgl, &rad_compute_shader, "reduce_main", "rad_reduce");
        let rad_finalize_pipeline = create_compute_pipeline(device, &rad_compute_bgl, &rad_compute_shader, "finalize_main", "rad_finalize");

        let make_rad_bg = |label, grid_read: &wgpu::Buffer, grid_write: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &rad_compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: rad_params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: igrf_coeffs_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: rad_aep8_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: grid_read.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: grid_write.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: rad_max_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&rad_output_view) },
                ],
            })
        };
        let rad_bg_ab = make_rad_bg("rad_bg_ab", &rad_grid_a, &rad_grid_b);
        let rad_bg_ba = make_rad_bg("rad_bg_ba", &rad_grid_b, &rad_grid_a);

        Self {
            planet_pipeline,
            rtt_pipeline,
            sun_pipeline,
            heatmap_pipeline,
            map_pipeline,
            planet_bgl,
            _sun_bgl: sun_bgl,
            heatmap_bgl,
            map_bgl,
            sampler_repeat,
            sampler_clamp,
            textures: HashMap::new(),
            cloud_textures: HashMap::new(),
            night_texture: None,
            star_texture: None,
            milky_way_texture: None,
            ring_textures: HashMap::new(),
            heatmap_palette_view,
            igrf_coeffs_buf,
            heatmap_data_view: None,
            heatmap_data_key: 0,
            map_texture_view: None,
            map_texture_key: None,
            detail_view: None,
            dummy_view,
            _dummy_view_rgba: dummy_view_rgba,
            _target_format: format,
            sun_ub,
            planet_bg_queue: Vec::new(), planet_bg_paint_idx: std::sync::atomic::AtomicUsize::new(0),
            heatmap_bg_queue: Vec::new(), heatmap_bg_paint_idx: std::sync::atomic::AtomicUsize::new(0),
            map_bg_queue: Vec::new(), map_bg_paint_idx: std::sync::atomic::AtomicUsize::new(0),
            sun_bg,
            texture_gen: 0,
            rad_trace_pipeline, rad_blur_pipeline, rad_reduce_pipeline, rad_finalize_pipeline,
            rad_compute_bgl, rad_bg_ab, rad_bg_ba,
            rad_params_buf, rad_aep8_buf, rad_grid_a, rad_grid_b, rad_max_buf,
            rad_output_tex, rad_output_view, rad_computed_params: None,
        }
    }

    pub fn upload_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: (CelestialBody, Skin, TextureResolution), earth_tex: &EarthTexture) {
        if self.textures.contains_key(&key) { return; }
        let pixels: Vec<u8> = earth_tex.pixels.iter().flat_map(|&[r, g, b]| [r, g, b]).collect();
        let view = upload_rgb_texture(device, queue, earth_tex.width as u32, earth_tex.height as u32, &pixels);
        self.textures.insert(key, view);
        self.texture_gen += 1;
    }

    pub fn upload_cloud_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, res: TextureResolution, cloud_tex: &EarthTexture) {
        if self.cloud_textures.contains_key(&res) { return; }
        let pixels: Vec<u8> = cloud_tex.pixels.iter().flat_map(|&[r, g, b]| [r, g, b]).collect();
        let view = upload_rgb_texture(device, queue, cloud_tex.width as u32, cloud_tex.height as u32, &pixels);
        self.cloud_textures.insert(res, view);
        self.texture_gen += 1;
    }

    pub fn upload_night_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, tex: &EarthTexture) {
        if self.night_texture.is_some() { return; }
        let pixels: Vec<u8> = tex.pixels.iter().flat_map(|&[r, g, b]| [r, g, b]).collect();
        self.night_texture = Some(upload_rgb_texture(device, queue, tex.width as u32, tex.height as u32, &pixels));
        self.texture_gen += 1;
    }

    pub fn upload_star_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, tex: &EarthTexture) {
        if self.star_texture.is_some() { return; }
        let pixels: Vec<u8> = tex.pixels.iter().flat_map(|&[r, g, b]| [r, g, b]).collect();
        self.star_texture = Some(upload_rgb_texture(device, queue, tex.width as u32, tex.height as u32, &pixels));
        self.texture_gen += 1;
    }

    pub fn upload_milky_way_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, tex: &EarthTexture) {
        if self.milky_way_texture.is_some() { return; }
        let pixels: Vec<u8> = tex.pixels.iter().flat_map(|&[r, g, b]| [r, g, b]).collect();
        self.milky_way_texture = Some(upload_rgb_texture(device, queue, tex.width as u32, tex.height as u32, &pixels));
        self.texture_gen += 1;
    }

    pub fn upload_ring_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, body: CelestialBody, tex: &RingTexture) {
        if self.ring_textures.contains_key(&body) { return; }
        let pixels: Vec<u8> = tex.pixels.iter().flat_map(|&[r, g, b, a]| [r, g, b, a]).collect();
        let view = upload_rgba_texture(device, queue, tex.width as u32, tex.height as u32, &pixels);
        self.ring_textures.insert(body, view);
        self.texture_gen += 1;
    }

    pub fn upload_heatmap_data(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: u64, data: &[u8], w: u32, h: u32) {
        if self.heatmap_data_key == key && self.heatmap_data_view.is_some() { return; }
        self.heatmap_data_view = Some(upload_rgba_texture(device, queue, w, h, data));
        self.heatmap_data_key = key;
        self.texture_gen += 1;
    }

    pub fn upload_map_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: (CelestialBody, Skin, TextureResolution), earth_tex: &EarthTexture) {
        if self.map_texture_key == Some(key) && self.map_texture_view.is_some() { return; }
        let pixels: Vec<u8> = earth_tex.pixels.iter().flat_map(|&[r, g, b]| [r, g, b]).collect();
        self.map_texture_view = Some(upload_rgb_texture(device, queue, earth_tex.width as u32, earth_tex.height as u32, &pixels));
        self.map_texture_key = Some(key);
        self.texture_gen += 1;
    }

    pub fn upload_detail_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, rgba: &[u8]) {
        self.detail_view = Some(upload_rgba_texture(device, queue, width, height, rgba));
        self.texture_gen += 1;
    }

    #[allow(dead_code)]
    pub fn invalidate_texture(&mut self, key: (CelestialBody, Skin, TextureResolution)) {
        self.textures.remove(&key);
    }

    pub fn evict_unused_textures(&mut self, keep: &[(CelestialBody, Skin, TextureResolution)]) {
        let before = self.textures.len() + self.ring_textures.len();
        self.textures.retain(|k, _| keep.contains(k));
        let keep_bodies: std::collections::HashSet<CelestialBody> = keep.iter().map(|k| k.0).collect();
        self.ring_textures.retain(|b, _| keep_bodies.contains(b));
        if self.textures.len() + self.ring_textures.len() != before {
            self.texture_gen += 1;
        }
    }

    #[allow(dead_code)]
    pub fn invalidate_map_texture(&mut self, key: (CelestialBody, Skin, TextureResolution)) {
        if self.map_texture_key == Some(key) {
            self.map_texture_view = None;
            self.map_texture_key = None;
        }
    }

    fn planet_view(&self, key: &(CelestialBody, Skin, TextureResolution)) -> &wgpu::TextureView {
        self.textures.get(key).unwrap_or(&self.dummy_view)
    }

    fn cloud_view(&self, res: TextureResolution) -> &wgpu::TextureView {
        self.cloud_textures.get(&res).unwrap_or(&self.dummy_view)
    }

    fn night_view(&self) -> &wgpu::TextureView {
        self.night_texture.as_ref().unwrap_or(&self.dummy_view)
    }

    fn star_view(&self, show_milky_way: bool) -> &wgpu::TextureView {
        if show_milky_way {
            self.milky_way_texture.as_ref().unwrap_or(&self.dummy_view)
        } else {
            self.star_texture.as_ref().unwrap_or(&self.dummy_view)
        }
    }

    fn ring_view(&self, body: CelestialBody) -> &wgpu::TextureView {
        self.ring_textures.get(&body).unwrap_or(&self.dummy_view)
    }

    fn detail_view_or_dummy(&self) -> &wgpu::TextureView {
        self.detail_view.as_ref().unwrap_or(&self.dummy_view)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_batch_to_images(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        requests: &[RttRequest],
    ) -> Vec<(CelestialBody, egui::ColorImage)> {
        if requests.is_empty() { return Vec::new(); }

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bpp = 4u32;
        let format = wgpu::TextureFormat::Rgba8Unorm;

        struct Item {
            body: CelestialBody,
            fbo_size: u32,
            padded_bpr: u32,
            render_tex: wgpu::Texture,
            readback: wgpu::Buffer,
            bg: wgpu::BindGroup,
        }

        let mut items: Vec<Item> = Vec::with_capacity(requests.len());

        for req in requests {
            if !self.textures.contains_key(&req.key) { continue; }

            let ring_params = if req.skip_rings { None } else { req.key.0.ring_params() };
            let outer_ratio = ring_params.map(|(_, _, o)| o as f64).unwrap_or(1.0);
            let img_scale = if outer_ratio > 1.0 { outer_ratio } else { 1.0 };
            let fbo_size = (req.size as f64 * img_scale).ceil() as u32;
            let scale = (req.size as f32) / (fbo_size as f32);

            let render_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("rtt"),
                size: wgpu::Extent3d { width: fbo_size, height: fbo_size, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });

            let rot_cols = mat3_to_padded_cols(&req.inv_rotation);
            let (ring_inner, ring_outer) = ring_params.map(|(_, i, o)| (i, o)).unwrap_or((0.0, 0.0));
            let has_rings_f = if self.ring_textures.contains_key(&req.key.0) { 1.0f32 } else { 0.0 };
            let adams = if has_rings_f > 0.5 && req.key.0 == CelestialBody::Neptune { 1.0f32 } else { 0.0 };
            let eps = if has_rings_f > 0.5 && req.key.0 == CelestialBody::Uranus { 1.0f32 } else { 0.0 };

            let uniforms = PlanetUniforms {
                inv_rot_0: rot_cols[0], inv_rot_1: rot_cols[1], inv_rot_2: rot_cols[2],
                star_rot_0: rot_cols[0], star_rot_1: rot_cols[1], star_rot_2: rot_cols[2],
                sun_dir_flat: [0.0, 0.0, -1.0, req.flattening as f32],
                bg_aspect: [0.0, 0.0, 0.0, 1.0],
                detail_bounds: [0.0; 4],
                uv_etc: [1.0, 1.0, scale, 0.0],
                flags_a: [0.0, 0.0, 0.0, 0.0],
                flags_b: [0.0, 1.0, has_rings_f, ring_inner],
                flags_c: [ring_outer, adams, eps, 0.0],
            };

            let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rtt_ub"), size: std::mem::size_of::<PlanetUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&uniforms));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rtt_bg"),
                layout: &self.planet_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_repeat) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler_clamp) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(self.planet_view(&req.key)) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.dummy_view) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.dummy_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.dummy_view) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.dummy_view) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(self.ring_view(req.key.0)) },
                ],
            });

            let unpadded_bpr = fbo_size * bpp;
            let padded_bpr = (unpadded_bpr + align - 1) / align * align;

            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rtt_readback"),
                size: (padded_bpr * fbo_size) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            items.push(Item { body: req.key.0, fbo_size, padded_bpr, render_tex, readback, bg });
        }

        if items.is_empty() { return Vec::new(); }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rtt_batch") });

        let views: Vec<wgpu::TextureView> = items.iter()
            .map(|item| item.render_tex.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();

        for (i, item) in items.iter().enumerate() {
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("rtt"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &views[i],
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.rtt_pipeline);
                pass.set_bind_group(0, &item.bg, &[]);
                pass.draw(0..4, 0..1);
            }
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &item.render_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &item.readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(item.padded_bpr), rows_per_image: None } },
                wgpu::Extent3d { width: item.fbo_size, height: item.fbo_size, depth_or_array_layers: 1 },
            );
        }

        queue.submit(std::iter::once(encoder.finish()));

        for item in &items {
            let slice = item.readback.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
        }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(5)) });

        let mut results = Vec::with_capacity(items.len());
        for item in &items {
            let slice = item.readback.slice(..);
            let data = slice.get_mapped_range();
            let fs = item.fbo_size as usize;
            let mut pixels = Vec::with_capacity(fs * fs);
            for y in 0..fs {
                for x in 0..fs {
                    let offset = y * item.padded_bpr as usize + x * 4;
                    pixels.push(egui::Color32::from_rgba_unmultiplied(
                        data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                    ));
                }
            }
            drop(data);
            item.readback.unmap();
            results.push((item.body, egui::ColorImage { size: [fs, fs], pixels, source_size: egui::Vec2::ZERO }));
        }

        results
    }
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_sampler(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn bgl_texture(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    bgl: &wgpu::BindGroupLayout,
    wgsl: &str,
    label: &str,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        multiview: None,
        cache: None,
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub fn mat3_to_padded_cols(m: &Matrix3<f64>) -> [[f32; 4]; 3] {
    [
        [m[(0, 0)] as f32, m[(1, 0)] as f32, m[(2, 0)] as f32, 0.0],
        [m[(0, 1)] as f32, m[(1, 1)] as f32, m[(2, 1)] as f32, 0.0],
        [m[(0, 2)] as f32, m[(1, 2)] as f32, m[(2, 2)] as f32, 0.0],
    ]
}

pub struct PlanetPaintCallback {
    pub uniforms: PlanetUniforms,
    pub texture_key: (CelestialBody, Skin, TextureResolution),
    pub show_milky_way: bool,
    pub has_detail: bool,
}

impl PlanetPaintCallback {
    pub fn new(uniforms: PlanetUniforms, texture_key: (CelestialBody, Skin, TextureResolution), show_milky_way: bool, has_detail: bool) -> Self {
        Self { uniforms, texture_key, show_milky_way, has_detail }
    }
}

impl CallbackTrait for PlanetPaintCallback {
    fn prepare(&self, device: &wgpu::Device, queue: &wgpu::Queue, _sd: &ScreenDescriptor, _enc: &mut wgpu::CommandEncoder, res: &mut CallbackResources) -> Vec<wgpu::CommandBuffer> {
        let gpu = res.get_mut::<GpuResources>().unwrap();
        let paint_idx = gpu.planet_bg_paint_idx.load(std::sync::atomic::Ordering::Relaxed);
        if paint_idx >= gpu.planet_bg_queue.len() {
            gpu.planet_bg_queue.clear();
            gpu.planet_bg_paint_idx.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        let ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet_ub_inst"),
            size: std::mem::size_of::<PlanetUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ub, 0, bytemuck::bytes_of(&self.uniforms));
        let detail_v = if self.has_detail { gpu.detail_view_or_dummy() } else { &gpu.dummy_view };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet_bg"),
            layout: &gpu.planet_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&gpu.sampler_repeat) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&gpu.sampler_clamp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(gpu.planet_view(&self.texture_key)) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(gpu.cloud_view(self.texture_key.2)) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(gpu.night_view()) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(detail_v) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(gpu.star_view(self.show_milky_way)) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(gpu.ring_view(self.texture_key.0)) },
            ],
        });
        gpu.planet_bg_queue.push((ub, bg));
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, render_pass: &mut wgpu::RenderPass<'static>, res: &CallbackResources) {
        let gpu = res.get::<GpuResources>().unwrap();
        let idx = gpu.planet_bg_paint_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some((_, ref bg)) = gpu.planet_bg_queue.get(idx) {
            render_pass.set_pipeline(&gpu.planet_pipeline);
            render_pass.set_bind_group(0, bg, &[]);
            render_pass.draw(0..4, 0..1);
        }
    }
}

pub struct SunPaintCallback {
    pub uniforms: SunUniforms,
}

impl SunPaintCallback {
    pub fn new(uniforms: SunUniforms) -> Self {
        Self { uniforms }
    }
}

impl CallbackTrait for SunPaintCallback {
    fn prepare(&self, _device: &wgpu::Device, queue: &wgpu::Queue, _sd: &ScreenDescriptor, _enc: &mut wgpu::CommandEncoder, res: &mut CallbackResources) -> Vec<wgpu::CommandBuffer> {
        let gpu = res.get_mut::<GpuResources>().unwrap();
        queue.write_buffer(&gpu.sun_ub, 0, bytemuck::bytes_of(&self.uniforms));
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, render_pass: &mut wgpu::RenderPass<'static>, res: &CallbackResources) {
        let gpu = res.get::<GpuResources>().unwrap();
        if let Some(ref bg) = gpu.sun_bg {
            render_pass.set_pipeline(&gpu.sun_pipeline);
            render_pass.set_bind_group(0, bg, &[]);
            render_pass.draw(0..4, 0..1);
        }
    }
}

pub struct HeatmapPaintCallback {
    pub uniforms: HeatmapUniforms,
    pub heatmap_data: Option<(u64, Vec<u8>)>,
    pub compute_rad: Option<f32>,
}

impl HeatmapPaintCallback {
    pub fn new(uniforms: HeatmapUniforms, heatmap_data: Option<(u64, Vec<u8>)>, compute_rad: Option<f32>) -> Self {
        Self { uniforms, heatmap_data, compute_rad }
    }
}

impl CallbackTrait for HeatmapPaintCallback {
    fn prepare(&self, device: &wgpu::Device, queue: &wgpu::Queue, _sd: &ScreenDescriptor, _enc: &mut wgpu::CommandEncoder, res: &mut CallbackResources) -> Vec<wgpu::CommandBuffer> {
        let gpu = res.get_mut::<GpuResources>().unwrap();
        let mut cmds = Vec::new();

        if let Some(r_km) = self.compute_rad {
            let need = gpu.rad_computed_params.map(|p| p != r_km).unwrap_or(true);
            if need {
                queue.write_buffer(&gpu.rad_params_buf, 0, bytemuck::bytes_of(&[r_km, 0.0f32, 0.0f32, 0.0f32]));
                queue.write_buffer(&gpu.rad_max_buf, 0, &[0u8; 8]);

                let wgx = (181 + 15) / 16;
                let wgy = (91 + 15) / 16;

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rad_compute") });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rad_trace"), timestamp_writes: None });
                    pass.set_pipeline(&gpu.rad_trace_pipeline);
                    pass.set_bind_group(0, &gpu.rad_bg_ab, &[]);
                    pass.dispatch_workgroups(wgx, wgy, 1);
                }

                for i in 0..4u32 {
                    let bg = if i % 2 == 0 { &gpu.rad_bg_ab } else { &gpu.rad_bg_ba };
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rad_blur"), timestamp_writes: None });
                    pass.set_pipeline(&gpu.rad_blur_pipeline);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(wgx, wgy, 1);
                }

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rad_reduce"), timestamp_writes: None });
                    pass.set_pipeline(&gpu.rad_reduce_pipeline);
                    pass.set_bind_group(0, &gpu.rad_bg_ab, &[]);
                    pass.dispatch_workgroups(wgx, wgy, 1);
                }

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rad_finalize"), timestamp_writes: None });
                    pass.set_pipeline(&gpu.rad_finalize_pipeline);
                    pass.set_bind_group(0, &gpu.rad_bg_ab, &[]);
                    pass.dispatch_workgroups(wgx, wgy, 1);
                }

                gpu.rad_computed_params = Some(r_km);
                gpu.heatmap_data_view = Some(gpu.rad_output_tex.create_view(&wgpu::TextureViewDescriptor::default()));
                gpu.heatmap_data_key = r_km.to_bits() as u64;
                gpu.texture_gen += 1;
                cmds.push(encoder.finish());
            }
        } else if let Some((key, ref data)) = self.heatmap_data {
            gpu.upload_heatmap_data(device, queue, key, data, 181, 91);
        }

        let paint_idx = gpu.heatmap_bg_paint_idx.load(std::sync::atomic::Ordering::Relaxed);
        if paint_idx >= gpu.heatmap_bg_queue.len() {
            gpu.heatmap_bg_queue.clear();
            gpu.heatmap_bg_paint_idx.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        let ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("heatmap_ub_inst"),
            size: std::mem::size_of::<HeatmapUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ub, 0, bytemuck::bytes_of(&self.uniforms));
        let hd_view = gpu.heatmap_data_view.as_ref().unwrap_or(&gpu.dummy_view);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("heatmap_bg_inst"),
            layout: &gpu.heatmap_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&gpu.sampler_clamp) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gpu.heatmap_palette_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(hd_view) },
                wgpu::BindGroupEntry { binding: 4, resource: gpu.igrf_coeffs_buf.as_entire_binding() },
            ],
        });
        gpu.heatmap_bg_queue.push((ub, bg));
        cmds
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, render_pass: &mut wgpu::RenderPass<'static>, res: &CallbackResources) {
        let gpu = res.get::<GpuResources>().unwrap();
        let idx = gpu.heatmap_bg_paint_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some((_, ref bg)) = gpu.heatmap_bg_queue.get(idx) {
            render_pass.set_pipeline(&gpu.heatmap_pipeline);
            render_pass.set_bind_group(0, bg, &[]);
            render_pass.draw(0..4, 0..1);
        }
    }
}

pub struct MapPaintCallback {
    pub proj_shader_id: i32,
    pub shared_bounds: std::sync::Arc<egui::mutex::Mutex<[f64; 4]>>,
}

impl MapPaintCallback {
    pub fn new(proj_shader_id: i32, shared_bounds: std::sync::Arc<egui::mutex::Mutex<[f64; 4]>>) -> Self {
        Self { proj_shader_id, shared_bounds }
    }
}

impl CallbackTrait for MapPaintCallback {
    fn prepare(&self, device: &wgpu::Device, queue: &wgpu::Queue, _sd: &ScreenDescriptor, _enc: &mut wgpu::CommandEncoder, res: &mut CallbackResources) -> Vec<wgpu::CommandBuffer> {
        let gpu = res.get_mut::<GpuResources>().unwrap();
        // Reset the queue when the paint index has caught up (start of a new
        // frame). Otherwise multiple map views in the same frame would share
        // a single uniform buffer and all end up rendering with the last
        // projection that got prepared.
        let paint_idx = gpu.map_bg_paint_idx.load(std::sync::atomic::Ordering::Relaxed);
        if paint_idx >= gpu.map_bg_queue.len() {
            gpu.map_bg_queue.clear();
            gpu.map_bg_paint_idx.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        let [bx0, bx1, by0, by1] = *self.shared_bounds.lock();
        let (peirce, inv_scale) = if self.proj_shader_id == 10 {
            let c = crate::projection::peirce_const();
            ([c.m as f32, c.k_ as f32, c.big_k as f32, c.dx as f32], c.inv_scale as f32)
        } else {
            ([0.0; 4], 1.0)
        };
        let uniforms = MapUniforms {
            bounds: [bx0 as f32, bx1 as f32, by0 as f32, by1 as f32],
            peirce,
            proj_invscale: [self.proj_shader_id as f32, inv_scale, 0.0, 0.0],
        };
        let ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("map_ub_inst"),
            size: std::mem::size_of::<MapUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ub, 0, bytemuck::bytes_of(&uniforms));
        let earth_v = gpu.map_texture_view.as_ref().unwrap_or(&gpu.dummy_view);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("map_bg_inst"),
            layout: &gpu.map_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&gpu.sampler_repeat) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(earth_v) },
            ],
        });
        gpu.map_bg_queue.push((ub, bg));
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, render_pass: &mut wgpu::RenderPass<'static>, res: &CallbackResources) {
        let gpu = res.get::<GpuResources>().unwrap();
        let idx = gpu.map_bg_paint_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some((_, ref bg)) = gpu.map_bg_queue.get(idx) {
            render_pass.set_pipeline(&gpu.map_pipeline);
            render_pass.set_bind_group(0, bg, &[]);
            render_pass.draw(0..4, 0..1);
        }
    }
}


const PLANET_WGSL: &str = concat!(
"
struct Uniforms {
    inv_rot_0: vec4f, inv_rot_1: vec4f, inv_rot_2: vec4f,
    star_rot_0: vec4f, star_rot_1: vec4f, star_rot_2: vec4f,
    sun_dir_flat: vec4f,
    bg_aspect: vec4f,
    detail_bounds: vec4f,
    uv_etc: vec4f,
    flags_a: vec4f,
    flags_b: vec4f,
    flags_c: vec4f,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp_r: sampler;
@group(0) @binding(2) var samp_c: sampler;
@group(0) @binding(3) var t_planet: texture_2d<f32>;
@group(0) @binding(4) var t_clouds: texture_2d<f32>;
@group(0) @binding(5) var t_night: texture_2d<f32>;
@group(0) @binding(6) var t_detail: texture_2d<f32>;
@group(0) @binding(7) var t_stars: texture_2d<f32>;
@group(0) @binding(8) var t_ring: texture_2d<f32>;
",
"
struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2f, 4>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(1,1));
    var o: VsOut; o.uv = p[vi] * 0.5 + 0.5; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}
const PI: f32 = 3.14159265359;
const ATMO_COLOR: vec3f = vec3f(0.4, 0.7, 1.0);
const ATMO_THICKNESS: f32 = 0.06;

fn inv_rot() -> mat3x3f { return mat3x3f(u.inv_rot_0.xyz, u.inv_rot_1.xyz, u.inv_rot_2.xyz); }
fn star_rot() -> mat3x3f { return mat3x3f(u.star_rot_0.xyz, u.star_rot_1.xyz, u.star_rot_2.xyz); }

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let uv_scale = u.uv_etc.xy;
    let scale = u.uv_etc.z;
    let atmosphere = u.uv_etc.w;
    let aspect = u.bg_aspect.w;
    let bg_color = u.bg_aspect.xyz;
    let sun_dir = u.sun_dir_flat.xyz;
    let flattening = u.sun_dir_flat.w;
    let show_clouds = u.flags_a.x;
    let show_day_night = u.flags_a.y;
    let show_city_lights = u.flags_a.z;
    let show_stars = u.flags_a.w;
    let use_detail = u.flags_b.x;
    let transparent_bg = u.flags_b.y;
    let has_rings = u.flags_b.z;
    let ring_inner = u.flags_b.w;
    let ring_outer = u.flags_c.x;
    let adams_arc = u.flags_c.y;
    let epsilon_wobble = u.flags_c.z;

    var centered = (in.uv * uv_scale - 0.5) * 2.0;
    centered.x *= max(aspect, 1.0);
    centered.y *= max(1.0 / aspect, 1.0);
    centered /= scale;

    let b = 1.0 - flattening;
    let b2 = b * b;
    let ir = inv_rot();
    let O = ir * vec3f(centered.x, centered.y, 0.0);
    let D = ir * vec3f(0.0, 0.0, -1.0);
    let A = D.x*D.x + D.y*D.y/b2 + D.z*D.z;
    let B = 2.0 * (O.x*D.x + O.y*D.y/b2 + O.z*D.z);
    let C = O.x*O.x + O.y*O.y/b2 + O.z*O.z - 1.0;
    let discriminant = B*B - 4.0*A*C;
    let screen_dist = length(centered);
    let atmo_outer = 1.0 + ATMO_THICKNESS * atmosphere;

    var lat_ortho: f32 = 0.0;
    var lon_ortho: f32 = 0.0;
    var normal_ortho = vec3f(0.0, 0.0, 1.0);
    let ortho_hit = discriminant >= 0.0;

    if ortho_hit {
        let t = (-B - sqrt(discriminant)) / (2.0 * A);
        let wp = O + t * D;
        lat_ortho = asin(clamp(wp.y / b, -1.0, 1.0));
        lon_ortho = atan2(-wp.z, wp.x);
        normal_ortho = normalize(vec3f(wp.x, wp.y / b2, wp.z));
    }

    var t_sphere: f32 = 1e10;
    if ortho_hit { t_sphere = (-B - sqrt(discriminant)) / (2.0 * A); }

    var ring_alpha: f32 = 0.0;
    var ring_color = vec3f(0.0);
    var t_ring_hit: f32 = 1e10;
    if has_rings > 0.5 && abs(D.y) > 0.0001 {
        let td = -O.y / D.y;
        let rh = O + td * D;
        var r = length(vec2f(rh.x, rh.z));
        if epsilon_wobble > 0.5 {
            let theta = atan2(rh.z, rh.x);
            let eps_center: f32 = 2.017;
            let eps_zone: f32 = 0.06;
            let prox = 1.0 - smoothstep(0.0, eps_zone, abs(r - eps_center));
            if prox > 0.0 {
                let radial_shift = eps_center * 0.025 * cos(theta);
                r -= radial_shift * prox;
                let width_scale = 1.0 + 0.7 * cos(theta);
                let dr = r - eps_center;
                r = eps_center + dr / mix(1.0, width_scale, prox);
            }
        }
        if r >= ring_inner && r <= ring_outer {
            let ru = (r - ring_inner) / (ring_outer - ring_inner);
            let rs = textureSampleLevel(t_ring, samp_c, vec2f(ru, 0.5), 0.0);
            ring_color = rs.rgb;
            ring_alpha = rs.a;
            if adams_arc > 0.5 && ru > 0.82 {
                let ang = atan2(rh.z, rh.x);
                var deg = ang * 180.0 / PI;
                if deg < 0.0 { deg += 360.0; }
                var arc: f32 = 0.0;
                if deg > 237.0 && deg < 257.0 { arc = smoothstep(237.0, 240.0, deg) * (1.0 - smoothstep(254.0, 257.0, deg)); }
                if deg > 258.0 && deg < 265.0 { arc = max(arc, smoothstep(258.0, 260.0, deg) * (1.0 - smoothstep(263.0, 265.0, deg))); }
                if deg > 266.0 && deg < 273.0 { arc = max(arc, smoothstep(266.0, 268.0, deg) * (1.0 - smoothstep(271.0, 273.0, deg))); }
                if deg > 274.0 && deg < 290.0 { arc = max(arc, smoothstep(274.0, 277.0, deg) * (1.0 - smoothstep(287.0, 290.0, deg))); }
                if deg > 292.0 && deg < 320.0 { arc = max(arc, smoothstep(292.0, 296.0, deg) * (1.0 - smoothstep(316.0, 320.0, deg))); }
                ring_alpha *= mix(0.15, 1.0, arc);
            }
            t_ring_hit = td;
        }
    }

    let ring_in_front = ring_alpha > 0.01 && t_ring_hit < t_sphere;

    if !ortho_hit && ring_alpha < 0.01 {
        var bg = vec3f(0.0);
        var bg_a: f32 = 0.0;
        if show_stars > 0.5 {
            let vp_aspect = aspect * uv_scale.x / uv_scale.y;
            var sp = (in.uv - 0.5) * 2.0;
            sp.x *= vp_aspect;
            let dir = star_rot() * normalize(vec3f(sp, -2.0));
            let slat = asin(clamp(dir.y, -1.0, 1.0));
            let slon = atan2(-dir.z, dir.x);
            let su = (slon + PI) / (2.0 * PI);
            let sv = (PI / 2.0 - slat) / PI;
            bg = textureSampleLevel(t_stars, samp_r, vec2f(su, sv), 0.0).rgb;
            bg_a = 1.0;
        }
        if atmosphere > 0.0 && screen_dist < atmo_outer {
            let C_a = O.x*O.x + O.y*O.y + O.z*O.z - 1.0;
            let disc_a = B*B - 4.0*A*C_a;
            if disc_a >= 0.0 {
                var ad = (screen_dist - 1.0) / (ATMO_THICKNESS * atmosphere);
                ad = clamp(ad, 0.0, 1.0);
                var af = 1.0 - ad;
                af = pow(af, 2.0);
                let glow = af * 0.8;
                bg = bg * (1.0 - glow) + ATMO_COLOR * glow;
                bg_a = max(bg_a, glow);
            }
        }
        if transparent_bg > 0.5 { discard; }
        return vec4f(mix(bg_color, bg, bg_a), 1.0);
    }

    if !ortho_hit && ring_alpha >= 0.01 {
        var bg = vec3f(0.0);
        var bg_a: f32 = 0.0;
        if show_stars > 0.5 {
            let vp_aspect = aspect * uv_scale.x / uv_scale.y;
            var sp = (in.uv - 0.5) * 2.0;
            sp.x *= vp_aspect;
            let dir = star_rot() * normalize(vec3f(sp, -2.0));
            let slat = asin(clamp(dir.y, -1.0, 1.0));
            let slon = atan2(-dir.z, dir.x);
            let su = (slon + PI) / (2.0 * PI);
            let sv = (PI / 2.0 - slat) / PI;
            bg = textureSampleLevel(t_stars, samp_r, vec2f(su, sv), 0.0).rgb;
            bg_a = 1.0;
        }
        if transparent_bg > 0.5 { return vec4f(ring_color, ring_alpha); }
        let base = mix(bg_color, bg, bg_a);
        return vec4f(mix(base, ring_color, ring_alpha), 1.0);
    }

    let tex_u = (lon_ortho + PI) / (2.0 * PI);
    let tex_v = (PI / 2.0 - lat_ortho) / PI;

    var day_color: vec3f;
    if use_detail > 0.5 {
        var lon_deg = lon_ortho * 180.0 / PI;
        if lon_deg < u.detail_bounds.x { lon_deg += 360.0; }
        let du = (lon_deg - u.detail_bounds.x) / (u.detail_bounds.y - u.detail_bounds.x);
        let dv = (u.detail_bounds.w - lat_ortho) / (u.detail_bounds.w - u.detail_bounds.z);
        if du >= 0.0 && du <= 1.0 && dv >= 0.0 && dv <= 1.0 {
            day_color = textureSampleLevel(t_detail, samp_c, vec2f(du, dv), 0.0).rgb;
        } else {
            day_color = textureSampleLevel(t_planet, samp_r, vec2f(tex_u, tex_v), 0.0).rgb;
        }
    } else {
        day_color = textureSampleLevel(t_planet, samp_r, vec2f(tex_u, tex_v), 0.0).rgb;
    }

    if show_clouds > 0.5 && use_detail < 0.5 {
        let cloud = textureSampleLevel(t_clouds, samp_r, vec2f(tex_u, tex_v), 0.0).r;
        day_color = mix(day_color, vec3f(1.0), cloud);
    }

    var color: vec3f;
    let sun_dot = dot(normal_ortho, sun_dir);
    if show_day_night > 0.5 {
        let day_factor = smoothstep(-0.1, 0.1, sun_dot);
        let shade = 0.2 + 0.8 * max(sun_dot, 0.0);
        let lit_day = day_color * shade;
        var night_lights = vec3f(0.0);
        if show_city_lights > 0.5 {
            night_lights = textureSampleLevel(t_night, samp_r, vec2f(tex_u, tex_v), 0.0).rgb;
        }
        color = mix(night_lights, lit_day, day_factor);
    } else {
        let shade = 0.3 + 0.7 * max(dot(normal_ortho, -D), 0.0);
        color = day_color * shade;
    }

    if atmosphere > 0.0 {
        let fresnel = 1.0 - max(dot(normal_ortho, -D), 0.0);
        let fr3 = pow(fresnel, 3.0);
        let rim = fr3 * 0.6 * atmosphere;
        var atmo_sun: f32 = 1.0;
        if show_day_night > 0.5 { atmo_sun = max(sun_dot + 0.3, 0.0); }
        color = mix(color, ATMO_COLOR * atmo_sun, rim);
    }

    if ring_in_front { color = mix(color, ring_color, ring_alpha); }

    if transparent_bg > 0.5 { return vec4f(color, 1.0); }
    return vec4f(mix(bg_color, color, 1.0), 1.0);
}
");

const SUN_WGSL: &str = concat!(
"
struct Uniforms {
    uv_aspect_ps: vec4f,
    sun_cam_int: vec4f,
    zoom_pad: vec4f,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
",
"
struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2f, 4>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(1,1));
    var o: VsOut; o.uv = p[vi]; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}
const PI: f32 = 3.14159265359;

fn fmod(x: f32, y: f32) -> f32 { return x - y * floor(x / y); }

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let uv_scale = u.uv_aspect_ps.xy;
    let aspect = u.uv_aspect_ps.z;
    let ps = u.uv_aspect_ps.w;
    let sun_pos = u.sun_cam_int.xy;
    let cam_ratio = u.sun_cam_int.z;
    let intensity = u.sun_cam_int.w;
    let zoom_dilution = u.zoom_pad.x;

    let uv = in.uv * uv_scale;
    let av = vec2f(aspect, 1.0);
    let pos = uv * av;
    let sun = sun_pos * av;
    let rel = pos - sun;
    let d = length(rel);
    let planet_d = length(pos);

    let sun_r = ps * 0.72 * cam_ratio;
    let facing_sun = max(dot(normalize(pos + 0.001), normalize(sun + 0.001)), 0.0);
    let limb_shift = facing_sun * ps * 0.015;
    let core_mask = smoothstep(sun_r * 1.02, sun_r * 0.98, d);
    let core_vis = smoothstep(ps - limb_shift - ps * 0.01, ps - limb_shift + ps * 0.01, planet_d);
    let final_core = core_mask * core_vis * 3.0;

    let glow_ext = ps * 5.0 * cam_ratio;
    let nd = d / max(glow_ext, 0.001);
    let glow = exp(-nd * 3.5) * 0.8;
    let veil = pow(1.0 - clamp(nd, 0.0, 1.0), 4.0) * 0.1;
    let planet_mask = smoothstep(ps * 0.95, ps * 1.0, planet_d);
    let final_glow = (glow + veil) * planet_mask * intensity * zoom_dilution;

    let spike_angle = atan2(rel.y, rel.x);
    let spike_nd = d / max(glow_ext, 0.001);
    var spike_accum: f32 = 0.0;
    for (var i: i32 = 0; i < 8; i++) {
        let ray_a = f32(i) * 0.7854;
        let diff = abs(fmod(spike_angle - ray_a + 3.14159, 6.28318) - 3.14159);
        let needle = exp(-diff * 500.0) * exp(-spike_nd * 1.5);
        let halo = exp(-diff * 20.0) * 0.1 * exp(-spike_nd * 0.5);
        spike_accum += needle + halo;
    }
    let sun_dist = length(sun);
    let occ_factor = smoothstep(ps - sun_r, ps + sun_r, sun_dist);
    let final_spikes = spike_accum * 0.5 * occ_factor * intensity * zoom_dilution;

    var color = vec3f(final_core) + vec3f(1.0, 0.78, 0.3) * final_glow + vec3f(1.0, 0.85, 0.5) * final_spikes;

    let sun_n = normalize(sun + 0.001);
    let pos_n = normalize(pos + 0.001);
    let facing = max(dot(pos_n, sun_n), 0.0);
    let limb_d = abs(planet_d - ps) / (ps * 0.012);
    let ring_band = exp(-limb_d * limb_d);
    let on_outside = smoothstep(ps * 0.99, ps * 1.003, planet_d);
    let near_limb = smoothstep(ps * 1.03, ps * 1.003, planet_d);
    let eclipse_f = smoothstep(ps * 3.0, ps * 0.5, sun_dist);
    let ring_t = smoothstep(ps, ps * 1.02, planet_d);
    let ring_col = mix(vec3f(1.0, 0.3, 0.05), vec3f(0.2, 0.4, 1.0), ring_t);
    let ring = ring_band * on_outside * near_limb * facing * eclipse_f * 1.5 * intensity;
    color += ring_col * ring;

    return vec4f(max(color, vec3f(0.0)), 0.0);
}
");

const HEATMAP_WGSL: &str = concat!(
"
struct Uniforms {
    inv_rot_0: vec4f, inv_rot_1: vec4f, inv_rot_2: vec4f,
    mag_aspect: vec4f,
    dipole_scale: vec4f,
    uv_mode_smooth: vec4f,
    kp_pr_sr_sp: vec4f,
    se_pad: vec4f,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var t_palette: texture_2d<f32>;
@group(0) @binding(3) var t_data: texture_2d<f32>;
struct IgrfCoeffs { gc: array<f32, 104>, hc: array<f32, 104>, };
@group(0) @binding(4) var<storage, read> coeffs: IgrfCoeffs;
",
"
struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2f, 4>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(1,1));
    var o: VsOut; o.uv = p[vi] * 0.5 + 0.5; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}
const PI: f32 = 3.14159265359;
const A_KM: f32 = 6371.2;
const N_MAX: i32 = 8;


fn igrf_magnitude(pos_km: vec3f) -> f32 {
    let r = length(pos_km);
    let colat = acos(clamp(pos_km.y / r, -1.0, 1.0));
    let elon = atan2(-pos_km.z, pos_km.x);
    let ct = cos(colat);
    let st = max(sin(colat), 1e-6);
    let cp = cos(elon);
    let sp = sin(elon);

    var cos_m: array<f32, 9>;
    var sin_m: array<f32, 9>;
    cos_m[0] = 1.0; sin_m[0] = 0.0;
    for (var m: i32 = 1; m <= N_MAX; m++) {
        cos_m[m] = cos_m[m-1]*cp - sin_m[m-1]*sp;
        sin_m[m] = sin_m[m-1]*cp + cos_m[m-1]*sp;
    }

    var P: array<f32, 45>;
    var dP: array<f32, 45>;
    P[0] = 1.0;
    P[1] = ct; dP[1] = -st;
    P[2] = st; dP[2] = ct;

    for (var n: i32 = 2; n <= N_MAX; n++) {
        let nf = f32(n);
        let nn = n*(n+1)/2 + n;
        let nm1 = (n-1)*n/2 + n - 1;
        let sect = sqrt((2.0*nf - 1.0) / (2.0*nf));
        P[nn] = st * sect * P[nm1];
        dP[nn] = sect * (ct * P[nm1] + st * dP[nm1]);
        let nsub = n*(n+1)/2 + n - 1;
        let sd = sqrt(2.0*nf - 1.0);
        P[nsub] = ct * sd * P[nm1];
        dP[nsub] = sd * (-st * P[nm1] + ct * dP[nm1]);
        for (var m: i32 = 0; m < n - 1; m++) {
            let mf = f32(m);
            let denom = sqrt(nf*nf - mf*mf);
            let a = (2.0*nf - 1.0) / denom;
            let bb = sqrt((nf-1.0)*(nf-1.0) - mf*mf) / denom;
            let i0 = n*(n+1)/2 + m;
            let i1 = (n-1)*n/2 + m;
            let i2 = (n-2)*(n-1)/2 + m;
            P[i0] = a * ct * P[i1] - bb * P[i2];
            dP[i0] = a * (-st * P[i1] + ct * dP[i1]) - bb * dP[i2];
        }
    }

    let ratio = A_KM / r;
    var rp = ratio * ratio;
    var b_r: f32 = 0.0; var b_t: f32 = 0.0; var b_p: f32 = 0.0;
    for (var n: i32 = 1; n <= N_MAX; n++) {
        rp *= ratio;
        let nf1 = f32(n + 1);
        for (var m: i32 = 0; m <= n; m++) {
            let k = n*(n+1)/2 + m - 1;
            let pidx = n*(n+1)/2 + m;
            let ghp = coeffs.gc[k]*cos_m[m] + coeffs.hc[k]*sin_m[m];
            b_r += nf1 * rp * ghp * P[pidx];
            b_t -= rp * ghp * dP[pidx];
            if m > 0 {
                let mgh = f32(m) * (-coeffs.gc[k]*sin_m[m] + coeffs.hc[k]*cos_m[m]);
                b_p -= rp * mgh * P[pidx] / st;
            }
        }
    }
    return sqrt(b_r*b_r + b_t*b_t + b_p*b_p);
}

fn inv_rot() -> mat3x3f { return mat3x3f(u.inv_rot_0.xyz, u.inv_rot_1.xyz, u.inv_rot_2.xyz); }

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let uv_scale = u.uv_mode_smooth.xy;
    let mode = i32(u.uv_mode_smooth.z);
    let is_smooth = u.uv_mode_smooth.w;
    let aspect = u.mag_aspect.w;
    let scale = u.dipole_scale.w;
    let mag_axis = u.mag_aspect.xyz;
    let dipole_offset = u.dipole_scale.xyz;
    let kp = u.kp_pr_sr_sp.x;
    let planet_r = u.kp_pr_sr_sp.y;
    let sphere_r_km = u.kp_pr_sr_sp.z;
    let show_p = u.kp_pr_sr_sp.w;
    let show_e = u.se_pad.x;

    var centered = (in.uv * uv_scale - 0.5) * 2.0;
    centered.x *= max(aspect, 1.0);
    centered.y *= max(1.0 / aspect, 1.0);
    centered /= scale;

    let d_sq = dot(centered, centered);
    if d_sq > 1.0 { discard; }
    let d = sqrt(d_sq);
    let pz = sqrt(1.0 - d_sq);
    let gp = inv_rot() * vec3f(centered.x, centered.y, pz);

    let dp = gp - dipole_offset;
    let r_d = length(dp);
    let mag_dot = dot(dp, mag_axis);
    let sin_ml = mag_dot / r_d;
    let r_d_er = r_d / planet_r;

    var intensity: f32 = 0.0;

    if mode == 0 {
        let kp_scale = 0.5 + 0.5 * (kp / 9.0);
        let cos_ml_sq = 1.0 - sin_ml * sin_ml;
        var l: f32;
        if cos_ml_sq > 1e-6 { l = r_d_er / cos_ml_sq; } else { l = r_d_er * 1e6; }
        let inner_peak: f32 = 1.5;
        let inner_sigma: f32 = 0.3;
        let inner = exp(-(l - inner_peak) * (l - inner_peak) / (2.0 * inner_sigma * inner_sigma));
        let outer_peak: f32 = 4.5;
        let outer_sigma: f32 = 1.0;
        let outer = exp(-(l - outer_peak) * (l - outer_peak) / (2.0 * outer_sigma * outer_sigma));
        let belt = clamp((inner * 0.8 + outer * 1.0) * kp_scale, 0.0, 1.0);
        let r_c = length(gp);
        let saa_factor = pow(r_d / r_c, 12.0);
        intensity = clamp(belt * saa_factor, 0.0, 1.0);
    } else if mode == 1 {
        let b0: f32 = 30115.0;
        let f = b0 / pow(r_d_er, 3.0) * sqrt(1.0 + 3.0 * sin_ml * sin_ml);
        let ref_field = 30000.0 / pow(r_d_er, 3.0);
        let lo = ref_field * 0.67;
        let hi = ref_field * 2.2;
        intensity = clamp((f - lo) / (hi - lo), 0.0, 1.0);
    } else if mode == 2 {
        let pos_km = gp * sphere_r_km;
        let f = igrf_magnitude(pos_km);
        let r_er = sphere_r_km / A_KM;
        let ref_field = 30000.0 / (r_er * r_er * r_er);
        let lo = ref_field * 0.67;
        let hi = ref_field * 2.2;
        intensity = clamp((f - lo) / (hi - lo), 0.0, 1.0);
    } else if mode == 3 {
        let r_g = length(gp);
        let colat = acos(clamp(gp.y / r_g, -1.0, 1.0));
        let elon = atan2(-gp.z, gp.x);
        let u_tex = colat / PI;
        let v_tex = (elon + PI) / (2.0 * PI);
        let pe = textureSampleLevel(t_data, samp, vec2f(v_tex, u_tex), 0.0);
        let pval = pe.r * show_p;
        let eval = pe.g * show_e;
        intensity = max(pval, eval);
    }

    var t = clamp(intensity, 0.0, 1.0);
    if is_smooth < 0.5 { t = floor(t * 17.0) / 17.0; }
    let color = textureSampleLevel(t_palette, samp, vec2f(t, 0.5), 0.0).rgb;

    let edge_width = 3.0 * fwidth(d);
    let edge_alpha = smoothstep(1.0, 1.0 - edge_width, d);
    let alpha = edge_alpha * (180.0 / 255.0);
    if alpha < 0.001 { discard; }
    return vec4f(color, alpha);
}
");

const MAP_WGSL: &str = concat!(
"
struct Uniforms {
    bounds: vec4f,
    peirce: vec4f,
    proj_invscale: vec4f,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var t_earth: texture_2d<f32>;
",
"
struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2f, 4>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(1,1));
    var o: VsOut; o.uv = p[vi] * 0.5 + 0.5; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}
const PI: f32 = 3.14159265359;
const MOLL_S: f32 = 90.0;
const SIN_SX: f32 = 180.0;
const SIN_SY: f32 = 90.0;
const AE_S: f32 = 180.0;
const HAM_S: f32 = 90.0;
const CASS_S: f32 = 57.29577951;
const UTM_S: f32 = 57.29577951;
const LAEA_S: f32 = 90.0;
const GP_COS_STD: f32 = 0.7071067811865476;
const GP_SCALE_Y: f32 = 90.0;

fn ej(uu: f32, m: f32) -> vec3f {
    if m < 1e-15 { return vec3f(sin(uu), cos(uu), 1.0); }
    if m >= 1.0 - 1e-15 {
        let t = tanh(uu);
        let s = 1.0 / cosh(uu);
        return vec3f(t, s, s);
    }
    var aa: array<f32, 9>;
    var cc: array<f32, 9>;
    aa[0] = 1.0; cc[0] = sqrt(m);
    var b = sqrt(1.0 - m);
    var i: i32 = 0; var twon: f32 = 1.0;
    for (var iter: i32 = 0; iter < 8; iter++) {
        let ai = aa[i];
        if abs(cc[i] / ai) <= 1e-14 { break; }
        i++;
        cc[i] = (ai - b) / 2.0;
        aa[i] = (ai + b) / 2.0;
        b = sqrt(ai * b);
        twon *= 2.0;
    }
    var phi = twon * aa[i] * uu;
    for (var iter: i32 = 0; iter < 9; iter++) {
        let t2 = cc[i] * sin(phi) / aa[i];
        phi = (asin(clamp(t2, -1.0, 1.0)) + phi) / 2.0;
        i--;
        if i == 0 { break; }
    }
    let cn = cos(phi);
    let bp = phi;
    let dc = cos(phi - bp);
    var dn: f32;
    if abs(dc) > 1e-30 { dn = cn / dc; } else { dn = 1.0; }
    return vec3f(sin(phi), cn, dn);
}

fn eji(uu: f32, v: f32, m: f32, sn: ptr<function, vec2f>, cn: ptr<function, vec2f>, dn: ptr<function, vec2f>) {
    if abs(uu) < 1e-15 {
        let j = ej(v, 1.0 - m);
        *sn = vec2f(0.0, j.x / j.y);
        *cn = vec2f(1.0 / j.y, 0.0);
        *dn = vec2f(j.z / j.y, 0.0);
        return;
    }
    let ja = ej(uu, m);
    if abs(v) < 1e-15 {
        *sn = vec2f(ja.x, 0.0);
        *cn = vec2f(ja.y, 0.0);
        *dn = vec2f(ja.z, 0.0);
        return;
    }
    let jb = ej(v, 1.0 - m);
    let d = jb.y * jb.y + m * ja.x * ja.x * jb.x * jb.x;
    *sn = vec2f(ja.x * jb.z / d, ja.y * ja.z * jb.x * jb.y / d);
    *cn = vec2f(ja.y * jb.y / d, -ja.x * ja.z * jb.x * jb.z / d);
    *dn = vec2f(ja.z * jb.y * jb.z / d, -m * ja.x * ja.y * jb.x / d);
}

fn guyou_inv(x: f32, y: f32, k_: f32, m: f32, bk: f32) -> vec2f {
    var sn: vec2f; var cn: vec2f; var dn: vec2f;
    eji(0.5 * bk - y, -x, m, &sn, &cn, &dn);
    let d = cn.x * cn.x + cn.y * cn.y;
    if d < 1e-30 { return vec2f(0.0, 0.0); }
    let tr = (sn.x * cn.x + sn.y * cn.y) / d;
    let ti = (sn.y * cn.x - sn.x * cn.y) / d;
    let lam = -atan2(ti, tr);
    let la = k_ * (tr * tr + ti * ti);
    let phi = 2.0 * atan(exp(-0.5 * log(la))) - PI / 2.0;
    return vec2f(lam, phi);
}

fn inv_proj(p: i32, x: f32, y: f32) -> vec2f {
    if p == 0 {
        if abs(x) > 180.0 || abs(y) > 90.0 { return vec2f(-999.0); }
        return vec2f(y, x);
    }
    if p == 1 {
        let lr = 2.0*atan(exp(radians(y)))-PI/2.0;
        let ld = degrees(lr);
        if abs(ld)>85.0||abs(x)>180.0 { return vec2f(-999.0); }
        return vec2f(ld, x);
    }
    if p == 2 {
        let xn=x/MOLL_S; let yn=y/MOLL_S;
        let st=yn/sqrt(2.0);
        if abs(st)>1.0 { return vec2f(-999.0); }
        let th=asin(st); let ct=cos(th);
        if abs(ct)<1e-10 { return vec2f(-999.0); }
        let lon=PI*xn/(2.0*sqrt(2.0)*ct);
        if abs(lon)>PI { return vec2f(-999.0); }
        let sl=(2.0*th+sin(2.0*th))/PI;
        if abs(sl)>1.0 { return vec2f(-999.0); }
        return vec2f(degrees(asin(sl)),degrees(lon));
    }
    if p == 3 {
        let lr=y/SIN_SY*(PI/2.0);
        if abs(lr)>PI/2.0 { return vec2f(-999.0); }
        let cl=cos(lr);
        if abs(cl)<1e-10 { return vec2f(-999.0); }
        let lon=x/SIN_SX*PI/cl;
        if abs(lon)>PI { return vec2f(-999.0); }
        return vec2f(degrees(lr),degrees(lon));
    }
    if p == 4 {
        let xn=x/AE_S*PI; let yn=y/AE_S*PI;
        let c=sqrt(xn*xn+yn*yn);
        if c>PI { return vec2f(-999.0); }
        return vec2f(degrees(PI/2.0-c), degrees(atan2(xn,-yn)));
    }
    if p == 5 {
        let xn=x/HAM_S; let yn=y/HAM_S;
        let z2=1.0-(xn/4.0)*(xn/4.0)-(yn/2.0)*(yn/2.0);
        if z2<0.0 { return vec2f(-999.0); }
        let z=sqrt(z2);
        let lon=2.0*atan2(z*xn, 2.0*(2.0*z2-1.0));
        let sl=z*yn;
        if abs(sl)>1.0 { return vec2f(-999.0); }
        if abs(lon)>PI { return vec2f(-999.0); }
        return vec2f(degrees(asin(sl)), degrees(lon));
    }
    if p == 6 {
        let xr=radians(x); let yr=radians(y);
        if abs(yr)>PI/2.0 { return vec2f(-999.0); }
        if abs(yr)<=PI/4.0 {
            let sp=8.0*yr/(3.0*PI);
            if abs(sp)>1.0 { return vec2f(-999.0); }
            if abs(xr)>PI { return vec2f(-999.0); }
            return vec2f(degrees(asin(sp)), degrees(xr));
        }
        let sv=sign(yr);
        let sig=2.0-4.0*abs(yr)/PI;
        if sig<1e-10 { return vec2f(sv*90.0,0.0); }
        let step=PI/2.0;
        let lc=round((xr-PI/4.0)/step)*step+PI/4.0;
        let lam=lc+(xr-lc)/sig;
        if abs(lam-lc)>step/2.0+0.001 { return vec2f(-999.0); }
        if abs(lam)>PI+0.001 { return vec2f(-999.0); }
        let sp2=1.0-sig*sig/3.0;
        if sp2>1.0 { return vec2f(-999.0); }
        return vec2f(sv*degrees(asin(sp2)), clamp(degrees(lam),-180.0,180.0));
    }
    if p == 7 {
        let xr=x/CASS_S; let yr=y/CASS_S;
        if abs(xr)>PI/2.0 { return vec2f(-999.0); }
        return vec2f(degrees(asin(sin(yr)*cos(xr))), degrees(atan2(tan(xr),cos(yr))));
    }
    if p == 8 {
        let zone=clamp(floor((x+180.0)/6.0),0.0,59.0);
        let cm=zone*6.0-177.0;
        let xr=(x-cm)/UTM_S; let yr=y/UTM_S;
        let sl=sin(yr)/cosh(xr);
        if abs(sl)>1.0 { return vec2f(-999.0); }
        let dl=atan2(sinh(xr),cos(yr));
        if abs(dl)>radians(3.0)+0.001 { return vec2f(-999.0); }
        return vec2f(degrees(asin(sl)), cm+degrees(dl));
    }
    if p == 9 {
        let xn=x/LAEA_S; let yn=y/LAEA_S;
        let rho=sqrt(xn*xn+yn*yn);
        if rho>2.0 { return vec2f(-999.0); }
        if rho<1e-10 { return vec2f(0.0,0.0); }
        let c=2.0*asin(rho/2.0);
        return vec2f(degrees(asin(yn*sin(c)/rho)), degrees(atan2(xn*sin(c), rho*cos(c))));
    }
    if p == 11 {
        let lon = x / GP_COS_STD;
        if abs(lon)>180.0 { return vec2f(-999.0); }
        let sl = y / GP_SCALE_Y * GP_COS_STD;
        if abs(sl)>1.0 { return vec2f(-999.0); }
        return vec2f(degrees(asin(sl)), lon);
    }
    if p == 10 {
        let s12 = sqrt(0.5);
        let pm = u.peirce.x;
        let pk = u.peirce.y;
        let pbk = u.peirce.z;
        let pdx = u.peirce.w;
        let xx = x * u.proj_invscale.y;
        let yy = y * u.proj_invscale.y;
        let gx = (xx + yy) * s12;
        let gy = (yy - xx) * s12;
        let hd = 0.5 * pdx;
        let front = abs(gx)<hd+0.001 && abs(gy)<hd+0.001;
        if front {
            let ll = guyou_inv(gx, gy, pk, pm, pbk);
            let la = degrees(ll.y);
            let lo = degrees(ll.x);
            if abs(la)<=90.0 && abs(lo)<=180.0 { return vec2f(la, lo); }
            return vec2f(-999.0);
        }
        let dd = pdx * s12;
        var ss: f32;
        if (xx>0.0) != (yy>0.0) { ss = -1.0; } else { ss = 1.0; }
        var x1: f32; var y1: f32;
        if yy>0.0 { x1 = -ss*xx + dd; } else { x1 = -ss*xx - dd; }
        if xx>0.0 { y1 = -ss*yy + dd; } else { y1 = -ss*yy - dd; }
        let gx2 = (-x1-y1)*s12;
        let gy2 = (x1-y1)*s12;
        let ll = guyou_inv(gx2, gy2, pk, pm, pbk);
        var lo: f32;
        if gx2>0.0 { lo = degrees(ll.x) + 180.0; } else { lo = degrees(ll.x) - 180.0; }
        let la = degrees(ll.y);
        if abs(la)<=90.0 && abs(lo)<=180.0 { return vec2f(la, lo); }
        return vec2f(-999.0);
    }
    return vec2f(-999.0);
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let proj = i32(u.proj_invscale.x);
    let x = mix(u.bounds.x, u.bounds.y, in.uv.x);
    let y = mix(u.bounds.z, u.bounds.w, in.uv.y);
    let ll = inv_proj(proj, x, y);
    if ll.x < -900.0 { return vec4f(0.0); }
    let tu = (ll.y + 180.0) / 360.0;
    let tv = (90.0 - ll.x) / 180.0;
    return textureSampleLevel(t_earth, samp, vec2f(tu, tv), 0.0);
}
");

const RAD_COMPUTE_WGSL: &str = "
struct Params { r_km: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> params: Params;

struct IgrfCoeffs { gc: array<f32, 104>, hc: array<f32, 104> };
@group(0) @binding(1) var<storage, read> coeffs: IgrfCoeffs;

@group(0) @binding(2) var<storage, read> aep8: array<i32>;

@group(0) @binding(3) var<storage, read_write> grid_a: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> grid_b: array<vec2f>;

@group(0) @binding(5) var<storage, read_write> max_buf: array<atomic<u32>, 2>;

@group(0) @binding(6) var t_out: texture_storage_2d<rgba8unorm, write>;

const PI: f32 = 3.14159265359;
const A_KM: f32 = 6371.2;
const TRACE_N: i32 = 4;
const GRID_W: u32 = 181u;
const GRID_H: u32 = 91u;
const LOG10E: f32 = 0.4342945;
const AP8_DESC_OFF: i32 = 0;
const AE8_DESC_OFF: i32 = 8;
const AP8_MAP_OFF: i32 = 16;
const AE8_MAP_OFF: i32 = 16312;

fn igrf_field_vec(r_km: f32, colat: f32, elon: f32) -> vec3f {
    let st = max(sin(colat), 1e-6);
    let ct = cos(colat);
    let cp = cos(elon);
    let sp = sin(elon);

    var cos_m: array<f32, 5>;
    var sin_m: array<f32, 5>;
    cos_m[0] = 1.0; sin_m[0] = 0.0;
    for (var m = 1; m <= TRACE_N; m++) {
        cos_m[m] = cos_m[m-1]*cp - sin_m[m-1]*sp;
        sin_m[m] = sin_m[m-1]*cp + cos_m[m-1]*sp;
    }

    var P: array<f32, 15>;
    var dP: array<f32, 15>;
    P[0] = 1.0;
    P[1] = ct; dP[1] = -st;
    P[2] = st; dP[2] = ct;

    for (var n = 2; n <= TRACE_N; n++) {
        let nf = f32(n);
        let nn = n*(n+1)/2 + n;
        let nm1 = (n-1)*n/2 + n - 1;
        let sect = sqrt((2.0*nf - 1.0) / (2.0*nf));
        P[nn] = st * sect * P[nm1];
        dP[nn] = sect * (ct * P[nm1] + st * dP[nm1]);
        let nsub = n*(n+1)/2 + n - 1;
        let sd = sqrt(2.0*nf - 1.0);
        P[nsub] = ct * sd * P[nm1];
        dP[nsub] = sd * (-st * P[nm1] + ct * dP[nm1]);
        for (var m = 0; m < n - 1; m++) {
            let mf = f32(m);
            let denom = sqrt(nf*nf - mf*mf);
            let a = (2.0*nf - 1.0) / denom;
            let bb = sqrt((nf-1.0)*(nf-1.0) - mf*mf) / denom;
            let i0 = n*(n+1)/2 + m;
            let i1 = (n-1)*n/2 + m;
            let i2 = (n-2)*(n-1)/2 + m;
            P[i0] = a * ct * P[i1] - bb * P[i2];
            dP[i0] = a * (-st * P[i1] + ct * dP[i1]) - bb * dP[i2];
        }
    }

    let ratio = A_KM / r_km;
    var rp = ratio * ratio;
    var b_r: f32 = 0.0;
    var b_t: f32 = 0.0;
    var b_p: f32 = 0.0;
    for (var n = 1; n <= TRACE_N; n++) {
        rp *= ratio;
        let nf1 = f32(n + 1);
        for (var m = 0; m <= n; m++) {
            let k = n*(n+1)/2 + m - 1;
            let pidx = n*(n+1)/2 + m;
            let ghp = coeffs.gc[k]*cos_m[m] + coeffs.hc[k]*sin_m[m];
            b_r += nf1 * rp * ghp * P[pidx];
            b_t -= rp * ghp * dP[pidx];
            if m > 0 {
                let mgh = f32(m) * (-coeffs.gc[k]*sin_m[m] + coeffs.hc[k]*cos_m[m]);
                b_p -= rp * mgh * P[pidx] / st;
            }
        }
    }
    return vec3f(b_r, b_t, b_p);
}

fn trace_field_line(r_km: f32, colat: f32, elon: f32) -> vec2f {
    let b_loc = igrf_field_vec(r_km, colat, elon);
    let b_local = length(b_loc);
    var b_min = b_local;
    var r_max = r_km;

    for (var sgn = 0; sgn < 2; sgn++) {
        let sign_f = select(-1.0, 1.0, sgn == 0);
        var r = r_km;
        var theta = colat;
        var phi = elon;
        var r_prev = r;
        var r_prev2 = r;
        var local_max = r;
        var passed_apex = false;

        for (var s = 0; s < 2000; s++) {
            let bv = igrf_field_vec(r, theta, phi);
            let b_mag = length(bv);
            if b_mag < 1e-10 { break; }
            if b_mag < b_min { b_min = b_mag; }

            r_prev2 = r_prev;
            r_prev = r;
            let ds = sign_f * 50.0;
            let st = max(sin(theta), 1e-10);
            r += bv.x / b_mag * ds;
            theta += bv.y / (r * b_mag) * ds;
            phi += bv.z / (r * st * b_mag) * ds;
            theta = clamp(theta, 0.01, PI - 0.01);

            if r > local_max { local_max = r; }
            if !passed_apex && r < r_prev && r_prev >= r_prev2 {
                passed_apex = true;
                let a = r_prev2;
                let b = r_prev;
                let c = r;
                let apex = b + 0.125 * (a - c) * (a - c) / max(abs(a - 2.0*b + c), 1e-10);
                if apex > local_max { local_max = apex; }
            }
            if r < local_max - 100.0 { break; }
            if r < A_KM || r > 20.0 * A_KM { break; }
        }
        if local_max > r_max { r_max = local_max; }
    }

    let l = min(r_max / A_KM, 20.0);
    var bb0: f32 = 1.0;
    if b_min > 0.0 { bb0 = b_local / b_min; }
    return vec2f(l, bb0);
}

fn trara2_fn(ms: i32, il: i32, ib: i32, fistep: f32) -> f32 {
    var i1: i32 = 0;
    var i2: i32 = 0;
    var l1: i32 = 0;
    var l2: i32 = 0;
    let fnl = f32(il);
    let fnb = f32(ib);

    for (var iter = 0; iter < 200; iter++) {
        l2 = aep8[ms + i2];
        if aep8[ms + i2 + 1] > il { break; }
        i1 = i2;
        l1 = l2;
        i2 += l2;
        if i2 > 20000 { return 0.0; }
    }

    if l1 < 4 && l2 < 4 { return 0.0; }

    if aep8[ms + i2 + 2] > aep8[ms + i1 + 2] {
        let ti = i1; i1 = i2; i2 = ti;
        let tl = l1; l1 = l2; l2 = tl;
    }

    let fll1 = f32(aep8[ms + i1 + 1]);
    let fll2 = f32(aep8[ms + i2 + 1]);
    let dfl = (fnl - fll1) / (fll2 - fll1);
    var flog1 = f32(aep8[ms + i1 + 2]);
    var flog2 = f32(aep8[ms + i2 + 2]);
    var fkb1: f32 = 0.0;
    var fkb2: f32 = 0.0;

    var fincr2 = f32(aep8[ms + i2 + 3]);
    var flogm = flog1 + (flog2 - flog1) * dfl;
    var fkbm: f32 = 0.0;
    fkb2 += fincr2;
    flog2 -= fistep;
    var sl2 = flog2 / fkb2;
    var sl1: f32;

    if l1 < 4 {
        sl1 = -900000.0;
    } else {
        let fincr1 = f32(aep8[ms + i1 + 3]);
        fkb1 += fincr1;
        flog1 -= fistep;
        sl1 = flog1 / fkb1;
    }

    var j1: i32 = 4;
    var j2: i32 = 4;

    for (var iter = 0; iter < 200; iter++) {
        if sl1 < sl2 {
            if j2 > l2 { return 0.0; }
            fincr2 = f32(aep8[ms + i2 + j2 - 1]);
            let fkbj2 = (flog2 / fistep * fincr2 + fkb2) / (fincr2 / fistep * sl1 + 1.0);
            let fkb = fkb1 + (fkbj2 - fkb1) * dfl;
            let flog = fkb * sl1;
            if fkb >= fnb {
                if fkb < fkbm + 1e-10 { return 0.0; }
                return max(flogm + (flog - flogm) * ((fnb - fkbm) / (fkb - fkbm)), 0.0);
            }
            fkbm = fkb;
            flogm = flog;
            if j1 >= l1 { return 0.0; }
            j1 += 1;
            let fincr1 = f32(aep8[ms + i1 + j1 - 1]);
            flog1 -= fistep;
            fkb1 += fincr1;
            sl1 = flog1 / fkb1;
        } else {
            if j1 > l1 { return 0.0; }
            let fincr1 = f32(aep8[ms + i1 + j1 - 1]);
            let fkbj1 = (flog1 / fistep * fincr1 + fkb1) / (fincr1 / fistep * sl2 + 1.0);
            let fkb = fkbj1 + (fkb2 - fkbj1) * dfl;
            let flog = fkb * sl2;
            if fkb >= fnb {
                if fkb < fkbm + 1e-10 { return 0.0; }
                return max(flogm + (flog - flogm) * ((fnb - fkbm) / (fkb - fkbm)), 0.0);
            }
            fkbm = fkb;
            flogm = flog;
            if j2 >= l2 { return 0.0; }
            j2 += 1;
            fincr2 = f32(aep8[ms + i2 + j2 - 1]);
            flog2 -= fistep;
            fkb2 += fincr2;
            sl2 = flog2 / fkb2;
        }
    }
    return 0.0;
}

fn trara1_fn(model: i32, fl: f32, bb0: f32, energy: f32) -> f32 {
    let desc_off = select(AE8_DESC_OFF, AP8_DESC_OFF, model == 0);
    let map_off = select(AE8_MAP_OFF, AP8_MAP_OFF, model == 0);

    let fistep = f32(aep8[desc_off + 6]) / f32(aep8[desc_off + 1]);
    let escale = f32(aep8[desc_off + 3]);
    let fscale = f32(aep8[desc_off + 6]);
    let xnl = min(abs(fl), 15.6);
    let nl = i32(xnl * f32(aep8[desc_off + 4]));
    let bb0c = max(bb0, 1.0);
    let nb = i32((bb0c - 1.0) * f32(aep8[desc_off + 5]));

    var i0: i32 = 0;
    var i1: i32 = 0;
    var i2: i32 = aep8[map_off];
    var i3 = i2 + aep8[map_off + i2];
    var l3 = aep8[map_off + i3];
    var e0: f32 = 0.0;
    var e1 = f32(aep8[map_off + 1]) / escale;
    var e2 = f32(aep8[map_off + i2 + 1]) / escale;
    var f0: f32 = 1.001;
    var f1: f32 = 1.001;
    var f2: f32 = 1.002;
    var s1 = true;
    var s2 = true;
    var s0 = true;

    for (var iter = 0; iter < 100; iter++) {
        if energy <= e2 || l3 == 0 { break; }
        i0 = i1; i1 = i2; i2 = i3;
        i3 += l3;
        l3 = aep8[map_off + i3];
        e0 = e1; e1 = e2;
        e2 = f32(aep8[map_off + i2 + 1]) / escale;
        s0 = s1; s1 = s2; s2 = true;
        f0 = f1; f1 = f2;
    }

    if s1 { f1 = trara2_fn(map_off + i1 + 2, nl, nb, fistep) / fscale; }
    if s2 { f2 = trara2_fn(map_off + i2 + 2, nl, nb, fistep) / fscale; }

    var result = f1 + (f2 - f1) * (energy - e1) / (e2 - e1);

    if f2 <= 0.0 && i1 != 0 {
        if s0 { f0 = trara2_fn(map_off + i0 + 2, nl, nb, fistep) / fscale; }
        let alt = f0 + (f1 - f0) * (energy - e0) / (e1 - e0);
        result = min(result, alt);
    }

    return max(result, 0.0);
}

@compute @workgroup_size(16, 16)
fn trace_main(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= GRID_W || gid.y >= GRID_H { return; }
    let colat = PI * f32(gid.y) / f32(GRID_H - 1u);
    let elon = -PI + 2.0 * PI * f32(gid.x) / f32(GRID_W - 1u);
    let lb = trace_field_line(params.r_km, colat, elon);
    let p = trara1_fn(0, lb.x, lb.y, 10.0);
    let e = trara1_fn(1, lb.x, lb.y, 1.0);
    let idx = gid.y * GRID_W + gid.x;
    grid_a[idx] = vec2f(p, e);
}

@compute @workgroup_size(16, 16)
fn blur_main(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= GRID_W || gid.y >= GRID_H { return; }
    var sum = vec2f(0.0);
    var w: f32 = 0.0;
    for (var dc: i32 = -1; dc <= 1; dc++) {
        for (var dl: i32 = -1; dl <= 1; dl++) {
            let c = i32(gid.y) + dc;
            var l = i32(gid.x) + dl;
            if c < 0 || c >= i32(GRID_H) { continue; }
            l = ((l % i32(GRID_W)) + i32(GRID_W)) % i32(GRID_W);
            let k = select(select(1.0, 2.0, dc == 0 || dl == 0), 4.0, dc == 0 && dl == 0);
            sum += grid_a[u32(c) * GRID_W + u32(l)] * k;
            w += k;
        }
    }
    grid_b[gid.y * GRID_W + gid.x] = sum / w;
}

@compute @workgroup_size(16, 16)
fn reduce_main(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= GRID_W || gid.y >= GRID_H { return; }
    let v = grid_a[gid.y * GRID_W + gid.x];
    atomicMax(&max_buf[0], bitcast<u32>(v.x));
    atomicMax(&max_buf[1], bitcast<u32>(v.y));
}

@compute @workgroup_size(16, 16)
fn finalize_main(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= GRID_W || gid.y >= GRID_H { return; }
    let max_p = bitcast<f32>(atomicLoad(&max_buf[0]));
    let max_e = bitcast<f32>(atomicLoad(&max_buf[1]));
    let v = grid_a[gid.y * GRID_W + gid.x];
    var rr: f32 = 0.0;
    var gg: f32 = 0.0;
    if max_p > 0.0 { rr = v.x / max_p; }
    if max_e > 0.0 { gg = v.y / max_e; }
    textureStore(t_out, vec2i(i32(gid.x), i32(gid.y)), vec4f(rr, gg, 0.0, 1.0));
}
";
