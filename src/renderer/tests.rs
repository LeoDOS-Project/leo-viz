use super::*;
use crate::{
    math::lat_lon_to_matrix,
    tile::{tile_to_lon_lat, DetailBounds, TileCoord},
};
use egui::{Pos2, Rect};
use nalgebra::Vector3;

const SIZE: u32 = 1024;
const SCALE: f32 = 0.85;

const MAP_PROJECTIONS: [crate::projection::ProjectionKind; 12] = {
    use crate::projection::ProjectionKind::*;
    [
        Equirectangular,
        Mercator,
        Mollweide,
        Sinusoidal,
        AzimuthalEquidistant,
        Hammer,
        HEALPix,
        Cassini,
        TransverseMercator,
        LambertAzimuthalEqualArea,
        GallPeters,
        PeirceQuincuncial,
    ]
};

fn map_callback(
    projection: crate::projection::ProjectionKind,
    bounds: [f64; 4],
    detail: Option<DetailLayer>,
) -> MapPaintCallback {
    MapPaintCallback::new(
        egui::Id::new(("map-test", projection.shader_id())),
        projection,
        bounds,
        detail,
    )
}

fn pixel(frame: &[u8], x: u32, y: u32) -> [u8; 4] {
    frame[((y * SIZE + x) * 4) as usize..((y * SIZE + x) * 4 + 4) as usize]
        .try_into()
        .unwrap()
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn gpu_maps_align_esri_imagery_on_every_projection() {
    let mut harness = Harness::new();
    harness
        .resources
        .get_mut::<GpuResources>()
        .unwrap()
        .map_texture_view = Some(upload_rgb_texture(
        &harness.device,
        &harness.queue,
        1,
        1,
        &[20, 40, 180],
    ));
    for point in [[25.5, 48.7], [-179.0, 45.0]] {
        let atlas = if point[0] < -170.0 {
            DetailBounds {
                min_lon: 170.0,
                max_lon: 190.0,
                min_lat: 35_f64.to_radians(),
                max_lat: 55_f64.to_radians(),
            }
        } else {
            bounds(2, 0, 0, 4, 4)
        };
        harness.upload(&atlas, point, false);
        for projection in MAP_PROJECTIONS {
            let proj = projection.instance();
            let (x, y) = proj.project(point[1], point[0]).unwrap();
            let (lat, lon) = proj.inverse(x, y).unwrap();
            assert!(
                (lat - point[1]).abs() < 1e-5 && (lon - point[0]).abs() < 1e-5,
                "{} inverse at {point:?}: {lat},{lon}",
                projection.label()
            );
            let (xmin, xmax) = proj.x_range();
            let (ymin, ymax) = proj.y_range();
            for viewport in [
                [xmin, xmax, ymin, ymax],
                [x - 3.0, x + 3.0, y - 3.0, y + 3.0],
            ] {
                let cb = map_callback(
                    projection,
                    viewport,
                    Some(DetailLayer {
                        bounds: atlas.mercator_bounds(),
                        blend: 1.0,
                        ..Default::default()
                    }),
                );
                let frame = harness.render(&[cb], &[0]);
                let px = (((x - viewport[0]) / (viewport[1] - viewport[0]) * SIZE as f64) as u32)
                    .min(SIZE - 1);
                let py = (((viewport[3] - y) / (viewport[3] - viewport[2]) * SIZE as f64) as u32)
                    .min(SIZE - 1);
                let color = pixel(&frame, px, py);
                assert!(
                    color[0] > 180 && color[1] < 60 && color[2] < 60,
                    "{} at {point:?}: {color:?}",
                    projection.label()
                );
                if point[0] > 0.0 && viewport[0] == xmin {
                    if let Some(output) = std::env::var_os("LEO_VIZ_TILE_TEST_OUTPUT") {
                        image::save_buffer(
                            std::path::PathBuf::from(output)
                                .join(format!("esri-map-{}.png", projection.shader_id())),
                            &frame,
                            SIZE,
                            SIZE,
                            image::ColorType::Rgba8,
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn clipped_map_views_never_use_another_maps_camera() {
    use crate::projection::ProjectionKind;
    let mut harness = Harness::new();
    let atlas = bounds(2, 0, 0, 4, 4);
    harness.upload(&atlas, [25.5, 48.7], false);
    let detail = Some(DetailLayer {
        bounds: atlas.mercator_bounds(),
        blend: 1.0,
        ..Default::default()
    });
    let projection = ProjectionKind::Mercator;
    let (x, y) = projection.instance().project(48.7, 25.5).unwrap();
    for _ in 0..3 {
        let visible = map_callback(projection, [x - 0.1, x + 0.1, y - 0.1, y + 0.1], detail);
        let clipped = map_callback(
            ProjectionKind::Equirectangular,
            [-160.0, -150.0, 0.0, 10.0],
            detail,
        );
        let frame = harness.render(&[clipped, visible], &[1]);
        assert_eq!(
            pixel(&frame, SIZE / 2, SIZE / 2),
            [255, 0, 0, 255],
            "Clipped maps must not displace the visible map's imagery"
        );
    }
    let mut visible = map_callback(projection, [x - 0.1, x + 0.1, y - 0.1, y + 0.1], detail);
    visible.view_id = egui::Id::new("visible-map");
    let mut other = map_callback(projection, [-160.0, -150.0, 0.0, 10.0], detail);
    other.view_id = egui::Id::new("other-map");
    let callbacks = [other, visible];
    for order in [vec![1], vec![0, 1], vec![1, 0], vec![1]] {
        let frame = harness.render(&callbacks, &order);
        let expected = if order.last() == Some(&1) {
            [255, 0, 0, 255]
        } else {
            [0, 110, 30, 255]
        };
        assert_eq!(
            pixel(&frame, SIZE / 2, SIZE / 2),
            expected,
            "Map views with the same projection must retain their own cameras"
        );
    }
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn gpu_maps_preserve_fading_fallbacks_and_disabled_imagery() {
    let mut harness = Harness::new();
    let gpu = harness.resources.get_mut::<GpuResources>().unwrap();
    gpu.map_texture_view = Some(upload_rgb_texture(
        &harness.device,
        &harness.queue,
        1,
        1,
        &[20, 40, 180],
    ));
    gpu.detail_previous_view = Some(upload_rgb_texture(
        &harness.device,
        &harness.queue,
        1,
        1,
        &[0, 200, 0],
    ));
    let atlas = bounds(2, 0, 0, 4, 4);
    harness.upload(&atlas, [25.5, 48.7], false);
    for projection in MAP_PROJECTIONS {
        let (x, y) = projection.instance().project(48.7, 25.5).unwrap();
        let viewport = [x - 0.1, x + 0.1, y - 0.1, y + 0.1];
        let detail = DetailLayer {
            bounds: atlas.mercator_bounds(),
            previous_bounds: atlas.mercator_bounds(),
            blend: 0.5,
        };
        let frame = harness.render(&[map_callback(projection, viewport, Some(detail))], &[0]);
        let color = pixel(&frame, SIZE / 2, SIZE / 2);
        assert!(
            (color[0] as i32 - 128).abs() <= 1
                && (color[1] as i32 - 100).abs() <= 1
                && color[2] == 0,
            "{} fade: {color:?}",
            projection.label()
        );
        let frame = harness.render(&[map_callback(projection, viewport, None)], &[0]);
        assert_eq!(pixel(&frame, SIZE / 2, SIZE / 2), [20, 40, 180, 255]);
    }
    harness.upload(&atlas, [25.5, 48.7], true);
    for projection in MAP_PROJECTIONS {
        let (x, y) = projection.instance().project(48.7, 25.5).unwrap();
        let cb = map_callback(
            projection,
            [x - 0.1, x + 0.1, y - 0.1, y + 0.1],
            Some(DetailLayer {
                bounds: atlas.mercator_bounds(),
                blend: 1.0,
                ..Default::default()
            }),
        );
        let frame = harness.render(&[cb], &[0]);
        assert_eq!(pixel(&frame, SIZE / 2, SIZE / 2), [20, 40, 180, 255]);
    }
    for projection in MAP_PROJECTIONS {
        let (xmin, xmax) = projection.instance().x_range();
        let (_, ymax) = projection.instance().y_range();
        let frame = harness.render(
            &[map_callback(
                projection,
                [xmin, xmax, ymax + 1.0, ymax + 2.0],
                None,
            )],
            &[0],
        );
        assert!(
            frame.chunks_exact(4).all(|color| color[..3] == [0, 0, 0]),
            "{} must not repeat beyond its boundary",
            projection.label()
        );
    }
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn gpu_maps_keep_submetre_imagery_stable_at_level_23() {
    let mut harness = Harness::new();
    let tile = crate::tile::lon_lat_to_tile(25.5, 48.7, 23);
    let atlas = bounds(tile.z, tile.x - 2, tile.y - 2, 5, 5);
    let mut gradient = Vec::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            gradient.extend([
                (x * 255 / (SIZE - 1)) as u8,
                (y * 255 / (SIZE - 1)) as u8,
                0,
                255,
            ]);
        }
    }
    harness
        .resources
        .get_mut::<GpuResources>()
        .unwrap()
        .upload_detail_texture(&harness.device, &harness.queue, SIZE, SIZE, &gradient);
    for projection in MAP_PROJECTIONS {
        let proj = projection.instance();
        let (x, y) = proj.project(48.7, 25.5).unwrap();
        for shift in [0.0, 0.0000001, -0.0000001] {
            let viewport = [
                x - 0.00001,
                x + 0.00001,
                y - 0.00001 + shift,
                y + 0.00001 + shift,
            ];
            let detail = DetailLayer {
                bounds: atlas.mercator_bounds(),
                blend: 1.0,
                ..Default::default()
            };
            let uniforms = MapUniforms::new(projection, viewport, Some(detail));
            assert_eq!(uniforms.origin[2], 1.0, "{}", projection.label());
            let frame = harness.render(&[map_callback(projection, viewport, Some(detail))], &[0]);
            for row in [128, 384, 640, 896] {
                for col in [128, 384, 640, 896] {
                    let px = viewport[0]
                        + (viewport[1] - viewport[0]) * (col as f64 + 0.5) / SIZE as f64;
                    let py = viewport[3]
                        - (viewport[3] - viewport[2]) * (row as f64 + 0.5) / SIZE as f64;
                    let (lat, lon) = proj.inverse(px, py).unwrap();
                    let b = atlas.mercator_bounds();
                    let expected = [
                        (lon - b[0]) / (b[1] - b[0]),
                        (b[3] - lat.to_radians().sin().atanh()) / (b[3] - b[2]),
                    ];
                    let color = pixel(&frame, col, row);
                    for channel in 0..2 {
                        assert!(
                            (color[channel] as f64 - expected[channel] * 255.0).abs() < 2.0,
                            "{} shift {shift}, {col},{row}: {color:?} expected {expected:?}",
                            projection.label()
                        );
                    }
                }
            }
        }
    }
}

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    resources: CallbackResources,
    target: wgpu::Texture,
}

impl Harness {
    fn cached_atlas(
        &mut self,
        atlas: &mut crate::tile::DetailAtlas,
        tree: &mut crate::tile::TileQuadTree,
        needed: &[TileCoord],
        x: u32,
    ) -> Vec<DetailUploadStats> {
        let gpu = self.resources.get_mut::<GpuResources>().unwrap();
        let layout = crate::tile::AtlasLayout::new(needed, x);
        gpu.move_detail_atlas(&self.device, &self.queue, atlas.layout, layout);
        atlas.set_layout(layout);
        let mut stats = Vec::new();
        loop {
            let mut writer = gpu.detail_atlas_writer(&self.device, &self.queue, needed);
            let more = atlas.update_tiles(tree, needed, &mut writer);
            stats.push(writer.finish());
            if !more {
                break;
            }
        }
        assert!(atlas.covers(needed));
        gpu.publish_detail_atlas();
        stats
    }

    fn detail_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let gpu = self.resources.get::<GpuResources>().unwrap();
        let texture = gpu.detail_texture.as_ref().unwrap();
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let pixel = readback.slice(..).get_mapped_range()[..4]
            .try_into()
            .unwrap();
        readback.unmap();
        pixel
    }

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
    fn render<T: CallbackTrait>(&mut self, callbacks: &[T], paint: &[usize]) -> Vec<u8> {
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
            detail_previous_bounds: [0.0; 4],
            detail_blend: [1.0, 0.0, 0.0, 0.0],
            detail_center: [0.0; 4],
            detail_current_uv: [0.0; 4],
            detail_previous_uv: [0.0; 4],
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
        (
            [30.0_f64, 60.0],
            [bounds(3, 4, 0, 3, 4), bounds(4, 8, 2, 4, 6)],
        ),
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

#[test]
#[ignore = "requires a native GPU adapter"]
fn maximum_zoom_panning_matches_double_precision_mercator_coordinates() {
    for (level, zoom, step) in [
        (18, 20000.0, 0.000002),
        (23, crate::math::MAX_CAMERA_ZOOM, 0.00000004),
    ] {
        check_zoom_panning(level, zoom, step);
    }
}

fn check_zoom_panning(level: u8, zoom: f64, latitude_step: f64) {
    use crate::tile::{lon_lat_to_tile, AtlasLayout};
    let mut harness = Harness::new();
    let base_rotation = lat_lon_to_matrix(48.7_f64.to_radians(), 25.5_f64.to_radians());
    let base_inverse = base_rotation.transpose();
    let b = 1.0 - f64::from(0.00335281_f32);
    let d = base_inverse.column(2);
    let centre = d / (d.x * d.x + d.y * d.y / (b * b) + d.z * d.z).sqrt();
    let tile = lon_lat_to_tile(25.5, (centre.y / b).asin().to_degrees(), level);
    let layout = AtlasLayout {
        x: tile.x - 4,
        y: tile.y - 4,
        z: level,
        cols: 8,
        rows: 8,
    };
    let bounds = layout.bounds();
    let pixels: Vec<_> = (0..SIZE)
        .flat_map(|row| {
            std::iter::repeat_n([((row % 128) * 2) as u8, 0, 0, 255], SIZE as usize).flatten()
        })
        .collect();
    harness
        .resources
        .get_mut::<GpuResources>()
        .unwrap()
        .upload_detail_texture(&harness.device, &harness.queue, SIZE, SIZE, &pixels);
    let mut max_error = 0.0_f64;
    let mut legacy_error = 0.0_f64;
    for step in 0..12 {
        let rotation = lat_lon_to_matrix(
            (48.7 + step as f64 * latitude_step).to_radians(),
            25.5_f64.to_radians(),
        );
        let mut cb = callback("maximum_zoom", &rotation, &bounds);
        cb.uniforms.uv_etc[2] = (zoom / 1.15) as f32;
        let legacy_frame = harness.render(
            &[callback_with_scale(
                &rotation,
                &bounds,
                cb.uniforms.uv_etc[2],
            )],
            &[0],
        );
        let layer = DetailLayer {
            bounds: bounds.mercator_bounds(),
            ..Default::default()
        };
        [
            cb.uniforms.detail_center,
            cb.uniforms.detail_current_uv,
            cb.uniforms.detail_previous_uv,
        ] = layer.camera_coordinates(
            &rotation.transpose(),
            f64::from(cb.uniforms.sun_dir_flat[3]),
        );
        let scale = f64::from(cb.uniforms.uv_etc[2]);
        let b = 1.0 - f64::from(cb.uniforms.sun_dir_flat[3]);
        let frame = harness.render(&[cb], &[0]);
        let inverse = rotation.transpose();
        let direction = inverse * Vector3::new(0.0, 0.0, -1.0);
        for row in (100..924).step_by(9) {
            let origin = inverse
                * Vector3::new(
                    1.0 / SIZE as f64 / scale,
                    (1.0 - 2.0 * (row as f64 + 0.5) / SIZE as f64) / scale,
                    0.0,
                );
            let dot =
                |a: Vector3<f64>, c: Vector3<f64>| a.x * c.x + a.y * c.y / (b * b) + a.z * c.z;
            let a = dot(direction, direction);
            let bb = 2.0 * dot(origin, direction);
            let c = dot(origin, origin) - 1.0;
            let point = origin + direction * ((-bb - (bb * bb - 4.0 * a * c).sqrt()) / (2.0 * a));
            let mercator = (point.y / b).asin().tan().asinh();
            let texel = (bounds.max_lat.tan().asinh() - mercator)
                / (bounds.max_lat.tan().asinh() - bounds.min_lat.tan().asinh())
                * SIZE as f64
                - 0.5;
            let phase = texel.rem_euclid(128.0);
            if !(5.0..120.0).contains(&phase) {
                continue;
            }
            let actual = frame[((row * SIZE + SIZE / 2) * 4) as usize] as f64;
            let legacy = legacy_frame[((row * SIZE + SIZE / 2) * 4) as usize] as f64;
            max_error = max_error.max((actual - phase * 2.0).abs());
            legacy_error = legacy_error.max((legacy - phase * 2.0).abs());
        }
    }
    eprintln!("Maximum-zoom Mercator sampling error: {max_error:.3}; legacy {legacy_error:.3} color levels");
    assert!(
        max_error <= 2.0,
        "Visible vertical wobble at maximum zoom: {max_error:.3}"
    );
}

fn callback_with_scale(
    rotation: &Matrix3<f64>,
    bounds: &DetailBounds,
    scale: f32,
) -> PlanetPaintCallback {
    let mut cb = callback("maximum_zoom_legacy", rotation, bounds);
    cb.uniforms.uv_etc[2] = scale;
    cb
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn parent_tile_resampling_is_bounded_without_reuploading_cached_imagery() {
    use crate::tile::{DetailAtlas, TileCacheEntry, TileQuadTree, TILE_RESAMPLES_PER_FRAME};
    let mut harness = Harness::new();
    let mut atlas = DetailAtlas::default();
    let mut tree = TileQuadTree::new(100);
    let parent = TileCoord { x: 1, y: 1, z: 2 };
    tree.insert(
        parent,
        TileCacheEntry {
            pixels: vec![[70, 100, 150]; 65536],
            width: 256,
            height: 256,
        },
    );
    harness.cached_atlas(&mut atlas, &mut tree, &[parent], parent.x);
    let needed: Vec<_> = (8..16)
        .flat_map(|y| (8..16).map(move |x| TileCoord { x, y, z: 5 }))
        .collect();
    let stats = harness.cached_atlas(&mut atlas, &mut tree, &needed, 8);
    assert_eq!(stats.len(), 2);
    assert!(stats.iter().all(
        |stats| stats.uploaded_tiles == 0 && stats.resampled_tiles <= TILE_RESAMPLES_PER_FRAME
    ));
    assert_eq!(
        stats
            .iter()
            .map(|stats| stats.resampled_tiles)
            .sum::<usize>(),
        64
    );
    assert_eq!(harness.detail_pixel(0, 0), [70, 100, 150, 255]);
    assert_eq!(harness.detail_pixel(2047, 2047), [70, 100, 150, 255]);
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn geographic_gpu_tiles_survive_zoom_changes_and_dateline_panning() {
    use crate::tile::{AtlasLayout, DetailAtlas, TileCacheEntry, TileQuadTree, TILE_SIZE};
    let mut harness = Harness::new();
    let mut atlas = DetailAtlas::default();
    let mut tree = TileQuadTree::new(1024);
    let a = AtlasLayout {
        x: 7,
        y: 2,
        z: 4,
        cols: 4,
        rows: 4,
    };
    let b = AtlasLayout {
        x: 17,
        y: 7,
        z: 5,
        ..a
    };
    let c = AtlasLayout {
        x: 32,
        y: 16,
        z: 6,
        ..a
    };
    let wrap = AtlasLayout { x: 15, y: 2, ..a };
    let adjacent = AtlasLayout { x: 0, ..wrap };
    let coords = |layout: AtlasLayout| {
        (0..layout.rows)
            .flat_map(move |dy| {
                (0..layout.cols).map(move |dx| TileCoord {
                    x: (layout.x + dx) % (1 << layout.z),
                    y: layout.y + dy,
                    z: layout.z,
                })
            })
            .collect::<Vec<_>>()
    };
    for layout in [a, b, c, wrap, adjacent] {
        for coord in coords(layout) {
            tree.insert(
                coord,
                TileCacheEntry {
                    pixels: vec![[coord.x as u8, coord.y as u8, coord.z * 10]; 65536],
                    width: TILE_SIZE,
                    height: TILE_SIZE,
                },
            );
        }
    }
    for (index, layout) in [a, b, c, a, b, c, wrap, adjacent, wrap]
        .into_iter()
        .enumerate()
    {
        let needed = coords(layout);
        let stats = harness.cached_atlas(&mut atlas, &mut tree, &needed, layout.x);
        if (3..6).contains(&index) {
            assert_eq!(
                stats.len(),
                1,
                "Warm level must not be throttled across frames"
            );
            assert_eq!(stats[0].uploaded_tiles, 0);
            assert_eq!(stats[0].reused_tiles, 16);
        }
        if index == 8 {
            assert_eq!(stats.len(), 1);
            assert_eq!(stats[0].uploaded_tiles, 0);
            assert_eq!(
                stats[0].reused_tiles, 4,
                "Reuse tiles outside the last atlas"
            );
        }
        for (i, coord) in needed.iter().enumerate() {
            let pixel = harness.detail_pixel(
                i as u32 % layout.cols * TILE_SIZE + 128,
                i as u32 / layout.cols * TILE_SIZE + 128,
            );
            assert_eq!(pixel, [coord.x as u8, coord.y as u8, coord.z * 10, 255]);
        }
    }
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn gpu_tile_cache_upgrades_parent_fallbacks_and_protects_needed_tiles_during_eviction() {
    use crate::tile::{DetailAtlas, TileCacheEntry, TileQuadTree, TILE_SIZE};
    let mut harness = Harness::new();
    harness
        .resources
        .get_mut::<GpuResources>()
        .unwrap()
        .detail_tiles
        .max_tiles = 2;
    let mut atlas = DetailAtlas::default();
    let mut tree = TileQuadTree::new(100);
    let a = TileCoord { x: 5, y: 6, z: 4 };
    let b = TileCoord { x: 7, ..a };
    let c = TileCoord { x: 6, ..a };
    let entry = |color| TileCacheEntry {
        pixels: vec![color; 65536],
        width: TILE_SIZE,
        height: TILE_SIZE,
    };
    let mut parent = entry([0; 3]);
    for y in 0..256usize {
        for x in 0..256usize {
            parent.pixels[y * 256 + x] = [x as u8, y as u8, 0];
        }
    }
    let parent_coord = TileCoord { x: 1, y: 1, z: 2 };
    tree.insert(parent_coord, parent);
    tree.insert(b, entry([70, 80, 90]));
    tree.insert(c, entry([100, 110, 120]));
    harness.cached_atlas(&mut atlas, &mut tree, &[parent_coord], parent_coord.x);
    let stats = harness.cached_atlas(&mut atlas, &mut tree, &[a], a.x);
    assert_eq!(
        stats[0].uploaded_tiles, 0,
        "A previously unvisited child uses its cached parent"
    );
    assert_eq!(stats[0].resampled_tiles, 1);
    assert_eq!(harness.detail_pixel(0, 0), [64, 128, 0, 255]);
    assert_eq!(harness.detail_pixel(255, 255), [127, 191, 0, 255]);
    harness.cached_atlas(&mut atlas, &mut tree, &[b], b.x);
    let stats = harness.cached_atlas(&mut atlas, &mut tree, &[a], a.x);
    assert_eq!(stats[0].uploaded_tiles, 0);
    assert_eq!(stats[0].reused_tiles, 1);
    assert_eq!(harness.detail_pixel(0, 0), [64, 128, 0, 255]);
    assert_eq!(harness.detail_pixel(255, 255), [127, 191, 0, 255]);

    tree.insert(a, entry([1, 2, 3]));
    let stats = harness.cached_atlas(&mut atlas, &mut tree, &[a], a.x);
    assert_eq!(
        stats[0].uploaded_tiles, 1,
        "Replace cached low-resolution fallback"
    );
    assert_eq!(harness.detail_pixel(128, 128), [1, 2, 3, 255]);

    // A is now older than B, but must survive eviction even when C is processed first.
    harness.cached_atlas(&mut atlas, &mut tree, &[b], b.x);
    let stats = harness.cached_atlas(&mut atlas, &mut tree, &[c, a], a.x);
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].uploaded_tiles, 1);
    assert_eq!(stats[0].reused_tiles, 1);
    assert_eq!(stats[0].evicted_tiles, 1);
    assert_eq!(stats[0].cached_tiles, 2);
    let cache = &harness
        .resources
        .get::<GpuResources>()
        .unwrap()
        .detail_tiles;
    assert!(cache.tiles.contains_key(&a));
    assert!(cache.tiles.contains_key(&c));
    assert!(!cache.tiles.contains_key(&b));
    assert_eq!(harness.detail_pixel(128, 128), [1, 2, 3, 255]);
    assert_eq!(harness.detail_pixel(384, 128), [100, 110, 120, 255]);
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn incremental_atlas_moves_preserve_pixels_and_tile_updates_reuse_bind_groups() {
    use crate::tile::{AtlasLayout, TILE_SIZE};
    let mut harness = Harness::new();
    for (point, x) in [([30.0_f64, 60.0], 7), ([-150.0, 60.0], 15)] {
        let mut layout = AtlasLayout {
            x,
            y: 2,
            z: 4,
            cols: 4,
            rows: 4,
        };
        harness.upload(&layout.bounds(), point, false);
        let rotation = lat_lon_to_matrix(point[1].to_radians(), point[0].to_radians());
        for (dx, y) in [(1, 3), (0, 2), (1, 3)] {
            let next = AtlasLayout {
                x: (x + dx) % 16,
                y,
                ..layout
            };
            harness
                .resources
                .get_mut::<GpuResources>()
                .unwrap()
                .move_detail_atlas(&harness.device, &harness.queue, Some(layout), next);
            harness
                .resources
                .get_mut::<GpuResources>()
                .unwrap()
                .publish_detail_atlas();
            layout = next;
            let pixels = harness.render(
                &[callback("incremental", &rotation, &layout.bounds())],
                &[0],
            );
            assert!(landmark(&pixels).distance(expected(point, &rotation)) < 2.0);
        }

        let tile = crate::tile::lon_lat_to_tile(point[0], point[1], layout.z);
        let origin = [
            ((tile.x + 16 - layout.x) % 16) * TILE_SIZE,
            (tile.y - layout.y) * TILE_SIZE,
        ];
        let pixels = vec![[0u8, 0, 240, 255]; (TILE_SIZE * TILE_SIZE) as usize];
        let gpu = harness.resources.get_mut::<GpuResources>().unwrap();
        let generation = gpu.texture_gen;
        gpu.write_detail_region(
            &harness.queue,
            origin,
            [TILE_SIZE; 2],
            bytemuck::cast_slice(&pixels),
        );
        assert_eq!(generation, gpu.texture_gen);
        let frame = harness.render(
            &[callback("incremental", &rotation, &layout.bounds())],
            &[0],
        );
        let p = expected(point, &rotation);
        let color = &frame[((p.y as u32 * SIZE + p.x as u32) * 4) as usize..][..3];
        assert!(color[2] > 100 && color[2] > color[0] * 2, "{color:?}");
        let gpu = harness.resources.get::<GpuResources>().unwrap();
        assert_eq!(generation, gpu.texture_gen);
        assert_eq!(gpu.planet_bind_groups.len(), 1);

        // Reusing a spare texture after a zoom change must clear unrelated imagery.
        let zoomed = AtlasLayout { z: 5, ..layout };
        harness
            .resources
            .get_mut::<GpuResources>()
            .unwrap()
            .move_detail_atlas(&harness.device, &harness.queue, Some(layout), zoomed);
        harness
            .resources
            .get_mut::<GpuResources>()
            .unwrap()
            .publish_detail_atlas();
        let frame = harness.render(
            &[callback("incremental", &rotation, &zoomed.bounds())],
            &[0],
        );
        assert!(
            frame.chunks_exact(4).all(|p| p[0] < 50),
            "Stale atlas imagery after a zoom change"
        );
    }
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn preparing_a_new_detail_level_keeps_the_published_imagery_unchanged() {
    use crate::tile::AtlasLayout;
    let mut harness = Harness::new();
    let layout = AtlasLayout {
        x: 7,
        y: 2,
        z: 4,
        cols: 4,
        rows: 4,
    };
    let point = [30.0_f64, 60.0_f64];
    let rotation = lat_lon_to_matrix(point[1].to_radians(), point[0].to_radians());
    harness.upload(&layout.bounds(), point, false);
    let reference = harness.render(&[callback("warming", &rotation, &layout.bounds())], &[0]);
    let generation = harness.resources.get::<GpuResources>().unwrap().texture_gen;
    let prepared = AtlasLayout {
        x: 17,
        y: 7,
        z: 5,
        cols: 4,
        rows: 4,
    };
    let gpu = harness.resources.get_mut::<GpuResources>().unwrap();
    gpu.move_detail_atlas(&harness.device, &harness.queue, Some(layout), prepared);
    gpu.write_detail_region(
        &harness.queue,
        [0, 0],
        [SIZE; 2],
        bytemuck::cast_slice(&vec![[0u8, 180, 0, 255]; (SIZE * SIZE) as usize]),
    );
    assert_eq!(generation, gpu.texture_gen);
    assert_eq!(
        reference,
        harness.render(&[callback("warming", &rotation, &layout.bounds())], &[0])
    );
    let next = AtlasLayout { x: 18, ..prepared };
    harness
        .resources
        .get_mut::<GpuResources>()
        .unwrap()
        .move_detail_atlas(&harness.device, &harness.queue, Some(prepared), next);
    assert_eq!(
        reference,
        harness.render(&[callback("warming", &rotation, &layout.bounds())], &[0])
    );
    harness
        .resources
        .get_mut::<GpuResources>()
        .unwrap()
        .publish_detail_atlas();
    let pixels = harness.render(&[callback("warming", &rotation, &next.bounds())], &[0]);
    let p = expected(point, &rotation);
    let color = &pixels[((p.y as u32 * SIZE + p.x as u32) * 4) as usize..][..3];
    assert!(
        color[1] > 100 && color[0] < 10 && color[2] < 10,
        "{color:?}"
    );
}

#[test]
#[ignore = "requires a native GPU adapter"]
fn tile_fades_blend_geographic_layers_and_fall_back_to_the_base_image() {
    use crate::tile::AtlasLayout;
    let mut harness = Harness::new();
    let old = AtlasLayout {
        x: 7,
        y: 2,
        z: 4,
        cols: 4,
        rows: 4,
    };
    let next = AtlasLayout {
        x: 17,
        y: 7,
        z: 5,
        cols: 4,
        rows: 4,
    };
    let point = [30.0_f64, 60.0_f64];
    harness.upload(&old.bounds(), point, false);
    let gpu = harness.resources.get_mut::<GpuResources>().unwrap();
    gpu.move_detail_atlas(&harness.device, &harness.queue, Some(old), next);
    gpu.write_detail_region(
        &harness.queue,
        [0, 0],
        [SIZE; 2],
        bytemuck::cast_slice(&vec![[0u8, 180, 0, 255]; (SIZE * SIZE) as usize]),
    );
    gpu.publish_detail_atlas();

    for lat in [60.0_f64, 61.0] {
        let rotation = lat_lon_to_matrix(lat.to_radians(), point[0].to_radians());
        for blend in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let mut cb = callback("fade", &rotation, &next.bounds());
            cb.uniforms.detail_previous_bounds = old.bounds().shader_bounds();
            cb.uniforms.detail_blend[0] = blend;
            let pixels = harness.render(&[cb], &[0]);
            let p = expected(point, &rotation);
            let color = &pixels[((p.y as u32 * SIZE + p.x as u32) * 4) as usize..][..3];
            let expected = [255.0 * (1.0 - blend), 180.0 * blend, 0.0];
            for (actual, expected) in color.iter().zip(expected) {
                assert!(
                    (*actual as f32 - expected).abs() < 2.0,
                    "lat={lat}, blend={blend}: {color:?}"
                );
            }
        }
    }

    // With no prior bounds, fade from the base image, never the dummy texture.
    let rotation = lat_lon_to_matrix(point[1].to_radians(), point[0].to_radians());
    for blend in [0.0_f32, 0.5, 1.0] {
        let mut cb = callback("first-fade", &rotation, &next.bounds());
        cb.uniforms.detail_blend[0] = blend;
        let pixels = harness.render(&[cb], &[0]);
        let p = expected(point, &rotation);
        let color = &pixels[((p.y as u32 * SIZE + p.x as u32) * 4) as usize..][..3];
        let expected = [
            20.0 * (1.0 - blend),
            40.0 + 140.0 * blend,
            180.0 * (1.0 - blend),
        ];
        for (actual, expected) in color.iter().zip(expected) {
            assert!(
                (*actual as f32 - expected).abs() < 2.0,
                "blend={blend}: {color:?}"
            );
        }
    }
}
