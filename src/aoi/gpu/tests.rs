use super::*;
use crate::aoi::preview::{Overlay, Preview};
use egui::{Pos2, Rect};
use egui_plot::PlotBounds;

const SIZE: u32 = 1024;
const RADIUS: f64 = 6371.0;

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
        eprintln!("AOI GPU: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let mut resources = CallbackResources::default();
        resources.insert(Resources::new(&device, wgpu::TextureFormat::Rgba8Unorm));
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("aoi_test_target"),
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

    fn render(&mut self, callbacks: &[PaintCallback], paint: &[usize], frame: Rect, ppp: f32) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let screen = ScreenDescriptor {
            size_in_pixels: [SIZE, SIZE],
            pixels_per_point: ppp,
        };
        for callback in callbacks {
            callback.prepare(
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
                    label: Some("aoi_test_pass"),
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
            pass.set_viewport(
                frame.min.x * ppp,
                frame.min.y * ppp,
                frame.width() * ppp,
                frame.height() * ppp,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(
                (frame.min.x * ppp) as u32,
                (frame.min.y * ppp) as u32,
                (frame.width() * ppp) as u32,
                (frame.height() * ppp) as u32,
            );
            for &i in paint {
                callbacks[i].paint(
                    egui::PaintCallbackInfo {
                        viewport: frame,
                        clip_rect: frame,
                        pixels_per_point: ppp,
                        screen_size_px: [SIZE, SIZE],
                    },
                    &mut pass,
                    &self.resources,
                );
            }
        }
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }

    fn pixels(&self) -> Vec<u8> {
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aoi_test_pixels"),
            size: u64::from(SIZE * SIZE * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
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
        let bytes = readback.slice(..).get_mapped_range().to_vec();
        readback.unmap();
        bytes
    }
}

fn preview(lon: f64, lat: f64, extent: f64) -> Arc<Preview> {
    Arc::new(Preview::from_json(&serde_json::to_vec(&serde_json::json!({
        "schema_version": 1, "config": {"count": 1, "seed": 42},
        "aois": [{"anchor_lon": lon, "anchor_lat": lat, "area_km2": 1000,
            "geometry": {"type": "Polygon", "coordinates": [[
                [lon-extent, lat-extent], [lon+extent, lat-extent],
                [lon+extent, lat+extent], [lon-extent, lat+extent], [lon-extent, lat-extent],
            ]]}}],
    })).unwrap()).unwrap())
}

fn transform(frame: Rect, zoom: f64) -> PlotTransform {
    PlotTransform::new(frame, PlotBounds::new_symmetrical(zoom), [false, false])
}

fn callback(
    preview: &Preview,
    rotation: &Matrix3<f64>,
    transform: &PlotTransform,
    id: &'static str,
) -> PaintCallback {
    PaintCallback::new(
        egui::Id::new(id),
        preview.gpu_geometry(),
        rotation,
        RADIUS,
        0.996647,
        0.8,
        transform,
    )
}

fn lit_bounds(pixels: &[u8]) -> Option<Rect> {
    let points: Vec<_> = pixels
        .chunks_exact(4)
        .enumerate()
        .filter_map(|(i, c)| {
            (c[0].max(c[1]).max(c[2]) > 5)
                .then(|| Pos2::new((i as u32 % SIZE) as f32, (i as u32 / SIZE) as f32))
        })
        .collect();
    (!points.is_empty()).then(|| Rect::from_points(&points))
}

#[test]
#[ignore = "Requires a native GPU"]
fn gpu_pixels_match_cpu_projection_and_clip_the_horizon() {
    let mut harness = Harness::new();
    for (lon, lat, extent, ppp, inset) in [
        (-90.0, 0.0, 4.0, 1.0, 0.0),
        (-2.0, 20.0, 4.0, 1.0, 32.0),
        (-90.0, 60.0, 0.01, 2.0, 0.0),
    ] {
        let preview = preview(lon, lat, extent);
        let limit = SIZE as f32 / ppp - inset;
        let frame = Rect::from_min_max(Pos2::new(inset, inset), Pos2::new(limit, limit));
        let transform = transform(frame, 7000.0);
        let rotation = Matrix3::identity();
        let cpu = Overlay::new(preview.clone(), rotation, RADIUS, 0.003353, 0.8).mesh(&transform);
        let expected =
            Rect::from_points(&cpu.vertices.iter().map(|v| v.pos * ppp).collect::<Vec<_>>());
        let visible = callback(&preview, &rotation, &transform, "visible");
        let hidden = callback(
            &preview,
            &crate::math::rotation_from_drag(std::f64::consts::PI, 0.0),
            &transform,
            "hidden",
        );
        harness.render(&[hidden, visible], &[1], frame, ppp);
        let actual = lit_bounds(&harness.pixels()).expect("GPU overlay must not be blank");
        assert!(
            (actual.min - expected.min).length() < 3.0,
            "{actual:?} vs {expected:?}"
        );
        assert!(
            (actual.max - expected.max).length() < 3.0,
            "{actual:?} vs {expected:?}"
        );
        assert!(frame.expand(1.0).contains(actual.min / ppp));
        assert!(frame.expand(1.0).contains(actual.max / ppp));
        let hidden = callback(
            &preview,
            &crate::math::rotation_from_drag(std::f64::consts::PI, 0.0),
            &transform,
            "visible",
        );
        harness.render(&[hidden], &[0], frame, ppp);
        if lon == -90.0 {
            assert!(lit_bounds(&harness.pixels()).is_none());
        }
    }
}

#[test]
#[ignore = "Requires a native GPU; benchmarks the bundled 50k AOIs"]
fn benchmark_fifty_thousand_aois_and_reuse_uploaded_geometry() {
    let preview = Arc::new(crate::aoi::presets::load_default_jobs().unwrap());
    let mut harness = Harness::new();
    let frame = Rect::from_min_size(Pos2::ZERO, egui::vec2(SIZE as f32, SIZE as f32));
    let geometry = preview.gpu_geometry();
    eprintln!(
        "50k AOIs: {} line instances, {} point instances, {:.1} MB",
        geometry.segments.len(),
        geometry.dots.len(),
        (geometry.segments.len() * std::mem::size_of::<Segment>()) as f64 / 1e6
    );
    for zoom in [7000.0, 1000.0, 250.0] {
        let transform = transform(frame, zoom);
        let rotation = |i: usize| {
            crate::math::lat_lon_to_matrix(
                50.0_f64.to_radians(),
                (10.0 + i as f64 * 0.05).to_radians(),
            )
        };
        harness.render(
            &[callback(&preview, &rotation(0), &transform, "benchmark")],
            &[0],
            frame,
            1.0,
        );
        let start = std::time::Instant::now();
        for i in 0..60 {
            let mesh =
                Overlay::new(preview.clone(), rotation(i), RADIUS, 0.003353, 0.8).mesh(&transform);
            std::hint::black_box(mesh);
        }
        let cpu_ms = start.elapsed().as_secs_f64() * 1000.0 / 60.0;
        let start = std::time::Instant::now();
        for i in 0..60 {
            let callback = callback(&preview, &rotation(i), &transform, "benchmark");
            harness.resources.get_mut::<Resources>().unwrap().prepare(
                &harness.device,
                &harness.queue,
                &callback,
                1.0,
            );
        }
        let prepare_ms = start.elapsed().as_secs_f64() * 1000.0 / 60.0;
        let start = std::time::Instant::now();
        for i in 0..60 {
            harness.render(
                &[callback(&preview, &rotation(i), &transform, "benchmark")],
                &[0],
                frame,
                1.0,
            );
        }
        let gpu_ms = start.elapsed().as_secs_f64() * 1000.0 / 60.0;
        eprintln!("50k AOIs zoom={zoom}: CPU geometry {cpu_ms:.3} ms/frame; GPU preparation {prepare_ms:.3} ms/frame; GPU render + wait {gpu_ms:.3} ms/frame");
        assert!(lit_bounds(&harness.pixels()).is_some());
        if let Some(dir) = std::env::var_os("LEO_VIZ_GPU_SCREENSHOT_DIR") {
            image::RgbaImage::from_raw(SIZE, SIZE, harness.pixels())
                .unwrap()
                .save(std::path::Path::new(&dir).join(format!("aoi-gpu-{zoom}.png")))
                .unwrap();
        }
    }
    assert!(Arc::ptr_eq(&geometry, &preview.gpu_geometry()));
    assert_eq!(
        harness.resources.get::<Resources>().unwrap().geometry.len(),
        1
    );
    assert_eq!(harness.resources.get::<Resources>().unwrap().views.len(), 1);
}

#[test]
#[ignore = "Requires a native GPU; generates and benchmarks the three environmental overlays"]
fn benchmark_three_thousand_environmental_aois() {
    use crate::aoi::{
        data,
        generate::{Environment, Job, Settings},
    };
    let mut previews = Vec::new();
    for kind in ["forest", "river", "coast"] {
        let start = std::time::Instant::now();
        let source = data::load(&data::Input::default(), kind).unwrap();
        let mut job = Job::new(
            Settings::Environment(Environment {
                kind,
                count: 1000,
                seed: 42,
                mean_area: 1000.0,
                size_ratio: 100.0,
                mean_length: 100.0,
                length_ratio: 20.0,
                buffer: 2.0,
                inland: 2.0,
                offshore: 1.0,
                flow: 10.0,
                compactness: 0.7,
                latitude: 70.0,
                tile_edge: 0.1,
            }),
            Some(source),
        )
        .unwrap();
        let preview = loop {
            if let Some(p) = job.step().unwrap() {
                break Arc::new(p);
            }
        };
        eprintln!(
            "{kind}: {} total LOD segments; generated in {:?}",
            preview.gpu_geometry().segments.len(),
            start.elapsed()
        );
        previews.push(preview);
    }
    let mut harness = Harness::new();
    let frame = Rect::from_min_size(Pos2::ZERO, egui::vec2(SIZE as f32, SIZE as f32));
    for zoom in [7000.0, 1000.0, 250.0] {
        let transform = transform(frame, zoom);
        let mut active = 0;
        for preview in &previews {
            let scale = transform.dpos_dvalue_x().abs() as f32 * RADIUS as f32;
            active += preview
                .gpu_geometry()
                .segments
                .iter()
                .filter(|s| scale > s.scales[0] && scale <= s.scales[1])
                .count();
        }
        let rotation = crate::math::lat_lon_to_matrix(48.7_f64.to_radians(), 25.5_f64.to_radians());
        let callbacks = previews
            .iter()
            .enumerate()
            .map(|(index, p)| {
                callback(
                    p,
                    &rotation,
                    &transform,
                    ["forest", "river", "coast"][index],
                )
            })
            .collect::<Vec<_>>();
        for _ in 0..3 {
            harness.render(&callbacks, &[0, 1, 2], frame, 1.0);
        }
        let mut reference = None;
        for all_lods in [true, false] {
            harness
                .resources
                .get_mut::<Resources>()
                .unwrap()
                .submit_all_lods = all_lods;
            let start = std::time::Instant::now();
            for _ in 0..30 {
                harness.render(&callbacks, &[0, 1, 2], frame, 1.0);
            }
            let pixels = harness.pixels();
            if all_lods {
                reference = Some(pixels);
            } else {
                assert_eq!(reference.as_ref().unwrap(), &pixels);
            }
            eprintln!("3,000 environmental AOIs zoom={zoom}, all_lods={all_lods}: {active} selected segments; GPU render + wait {:.3} ms/frame", start.elapsed().as_secs_f64() * 1000.0 / 30.0);
        }
    }
}
