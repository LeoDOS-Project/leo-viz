use super::*;
use crate::{
    math::lat_lon_to_matrix,
    tile::{tile_to_lon_lat, DetailBounds, TileCoord},
};
use egui::{Pos2, Rect};
use nalgebra::Vector3;

const SIZE: u32 = 1024;
const SCALE: f32 = 0.85;

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    resources: CallbackResources,
    target: wgpu::Texture,
}

impl Harness {
    fn new() -> Self {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        eprintln!("Planet GPU: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let mut gpu = GpuResources::new(&device, &queue, wgpu::TextureFormat::Rgba8Unorm);
        gpu.textures.insert(
            key(),
            upload_rgb_texture(&device, &queue, 1, 1, &[20, 40, 180]),
        );
        let mut resources = CallbackResources::default();
        resources.insert(gpu);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("planet_projection_test"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Self {
            device,
            queue,
            resources,
            target,
        }
    }
    fn upload(&mut self, bounds: &DetailBounds, point: [f64; 2], transparent: bool) {
        let north = bounds.max_lat.tan().asinh();
        let south = bounds.min_lat.tan().asinh();
        let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
        for row in 0..SIZE {
            let lat = (north - (north - south) * (row as f64 + 0.5) / SIZE as f64)
                .sinh()
                .atan()
                .to_degrees();
            for col in 0..SIZE {
                let lon = bounds.min_lon
                    + (bounds.max_lon - bounds.min_lon) * (col as f64 + 0.5) / SIZE as f64;
                let delta = (lon - point[0] + 180.0).rem_euclid(360.0) - 180.0;
                let pixel = if transparent {
                    [0, 0, 0, 0]
                } else if delta.abs() < 0.8 && (lat - point[1]).abs() < 0.8 {
                    [255, 0, 0, 255]
                } else {
                    [0, 110, 30, 255]
                };
                pixels.extend(pixel);
            }
        }
        self.resources
            .get_mut::<GpuResources>()
            .unwrap()
            .upload_detail_texture(&self.device, &self.queue, SIZE, SIZE, &pixels);
    }
    fn render(&mut self, callbacks: &[PlanetPaintCallback], paint: &[usize]) -> Vec<u8> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let screen = ScreenDescriptor {
            size_in_pixels: [SIZE, SIZE],
            pixels_per_point: 1.0,
        };
        for cb in callbacks {
            cb.prepare(
                &self.device,
                &self.queue,
                &screen,
                &mut encoder,
                &mut self.resources,
            );
        }
        let view = self.target.create_view(&Default::default());
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("planet_test_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            let frame = Rect::from_min_size(Pos2::ZERO, egui::vec2(SIZE as f32, SIZE as f32));
            for &i in paint {
                callbacks[i].paint(
                    egui::PaintCallbackInfo {
                        viewport: frame,
                        clip_rect: frame,
                        pixels_per_point: 1.0,
                        screen_size_px: [SIZE, SIZE],
                    },
                    &mut pass,
                    &self.resources,
                );
            }
        }
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(SIZE * SIZE * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * 4),
                    rows_per_image: Some(SIZE),
                },
            },
            self.target.size(),
        );
        self.queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let pixels = readback.slice(..).get_mapped_range().to_vec();
        readback.unmap();
        pixels
    }
}

fn key() -> (CelestialBody, Skin, TextureResolution) {
    (CelestialBody::Earth, Skin::Default, TextureResolution::R512)
}
fn bounds(z: u8, x: u32, y: u32, cols: u32, rows: u32) -> DetailBounds {
    let (west, north) = tile_to_lon_lat(&TileCoord { x, y, z });
    let (east, south) = tile_to_lon_lat(&TileCoord {
        x: x + cols,
        y: y + rows,
        z,
    });
    DetailBounds {
        min_lon: west,
        max_lon: east,
        min_lat: south.to_radians(),
        max_lat: north.to_radians(),
    }
}
fn callback(id: &str, rotation: &Matrix3<f64>, bounds: &DetailBounds) -> PlanetPaintCallback {
    let cols = mat3_to_padded_cols(&rotation.transpose());
    PlanetPaintCallback::new(
        egui::Id::new(id),
        PlanetUniforms {
            inv_rot_0: cols[0],
            inv_rot_1: cols[1],
            inv_rot_2: cols[2],
            star_rot_0: cols[0],
            star_rot_1: cols[1],
            star_rot_2: cols[2],
            sun_dir_flat: [0.0, 0.0, 1.0, 0.00335281],
            bg_aspect: [0.0, 0.0, 0.0, 1.0],
            detail_bounds: bounds.shader_bounds(),
            uv_etc: [1.0, 1.0, SCALE, 0.0],
            flags_a: [0.0; 4],
            flags_b: [1.0, 0.0, 0.0, 0.0],
            flags_c: [0.0; 4],
        },
        key(),
        false,
        true,
    )
}
fn expected(point: [f64; 2], rotation: &Matrix3<f64>) -> Pos2 {
    let (lon, lat) = (point[0].to_radians(), point[1].to_radians());
    let p = rotation
        * Vector3::new(
            lat.cos() * lon.cos(),
            lat.sin() * 0.99664719,
            -lat.cos() * lon.sin(),
        );
    Pos2::new(
        SIZE as f32 * (0.5 + p.x as f32 * SCALE * 0.5),
        SIZE as f32 * (0.5 - p.y as f32 * SCALE * 0.5),
    )
}
fn landmark(pixels: &[u8]) -> Pos2 {
    let mut points = Vec::new();
    for (i, p) in pixels.chunks_exact(4).enumerate() {
        if p[0] > 50
            && u32::from(p[0]) > u32::from(p[1]) * 2
            && u32::from(p[0]) > u32::from(p[2]) * 2
        {
            points.push(Pos2::new(
                (i as u32 % SIZE) as f32 + 0.5,
                (i as u32 / SIZE) as f32 + 0.5,
            ));
        }
    }
    assert!(
        !points.is_empty(),
        "Expected visible geographic registration marker"
    );
    let sum = points
        .iter()
        .fold(egui::Vec2::ZERO, |sum, p| sum + p.to_vec2());
    (sum / points.len() as f32).to_pos2()
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn mercator_atlas_landmarks_stay_registered_across_rebuilds_and_rotation() {
    let mut harness = Harness::new();
    for (point, atlases) in [
        ([30.0_f64, 60.0], [bounds(3, 4, 0, 3, 4), bounds(4, 8, 2, 4, 6)]),
        (
            [179.0, 60.0],
            [bounds(3, 7, 0, 2, 4), bounds(4, 15, 2, 2, 6)],
        ),
    ] {
        let mut centers = Vec::new();
        let rotation = lat_lon_to_matrix(60_f64.to_radians(), point[0].to_radians());
        for (index, atlas) in atlases.iter().enumerate() {
            harness.upload(atlas, point, false);
            let pixels = harness.render(&[callback("globe", &rotation, atlas)], &[0]);
            let center = landmark(&pixels);
            assert!(
                center.distance(expected(point, &rotation)) < 2.0,
                "{center:?} vs {:?}",
                expected(point, &rotation)
            );
            centers.push(center);
            if let Some(dir) = std::env::var_os("LEO_VIZ_TILE_SCREENSHOT_DIR") {
                let path = std::path::PathBuf::from(dir)
                    .join(format!("tiles-{}-{index}.png", point[0] as i32));
                image::save_buffer(path, &pixels, SIZE, SIZE, image::ColorType::Rgba8).unwrap();
            }
        }
        assert!(centers[0].distance(centers[1]) < 1.5);
        let rotation = lat_lon_to_matrix(55_f64.to_radians(), (point[0] + 10.0).to_radians());
        let pixels = harness.render(&[callback("globe", &rotation, &atlases[1])], &[0]);
        assert!(landmark(&pixels).distance(expected(point, &rotation)) < 2.0);
    }
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn clipped_views_do_not_reuse_stale_camera_uniforms_and_missing_tiles_show_base() {
    let mut harness = Harness::new();
    let atlas = bounds(3, 4, 0, 3, 4);
    let point = [30.0, 60.0];
    harness.upload(&atlas, point, false);
    let other = lat_lon_to_matrix(20_f64.to_radians(), 30_f64.to_radians());
    for lat in [60_f64, 58.0, 62.0, 55.0, 60.0] {
        let rotation = lat_lon_to_matrix(lat.to_radians(), 30_f64.to_radians());
        let pixels = harness.render(
            &[
                callback("visible", &rotation, &atlas),
                callback("clipped", &other, &atlas),
            ],
            &[0],
        );
        assert!(landmark(&pixels).distance(expected(point, &rotation)) < 2.0);
        assert_eq!(
            harness
                .resources
                .get::<GpuResources>()
                .unwrap()
                .planet_bind_groups
                .len(),
            2
        );
    }
    harness.upload(&atlas, point, true);
    let rotation = lat_lon_to_matrix(60_f64.to_radians(), 30_f64.to_radians());
    let pixels = harness.render(&[callback("visible", &rotation, &atlas)], &[0]);
    let center = &pixels[((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize..][..4];
    assert!(
        center[2] > center[0] * 2 && center[2] > center[1] * 2,
        "{center:?}"
    );
}
