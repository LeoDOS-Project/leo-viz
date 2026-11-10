//! Render cached AOI boundaries to Web Mercator for geographic image previews.
use super::*;
use crate::{aoi::preview::Preview, projection::ProjectionKind, tile::AtlasLayout};
use egui_wgpu::RenderState;

pub(crate) struct MapTexture {
    pub layout: AtlasLayout,
    pub id: egui::TextureId,
    texture: wgpu::Texture,
    layers: Vec<(Arc<Preview>, f32)>,
    render: RenderState,
}

impl MapTexture {
    fn new(render: &RenderState, layout: AtlasLayout, size: [u32; 2]) -> Self {
        let texture = render.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pushbroom_aois"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let id = render.renderer.write().register_native_texture(
            &render.device,
            &texture.create_view(&Default::default()),
            wgpu::FilterMode::Linear,
        );
        Self {
            layout,
            id,
            texture,
            layers: Vec::new(),
            render: render.clone(),
        }
    }
}

impl Drop for MapTexture {
    fn drop(&mut self) {
        self.render.renderer.write().free_texture(&self.id);
    }
}

pub(crate) struct MapRenderer {
    resources: CallbackResources,
}

impl MapRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let mut resources = CallbackResources::default();
        resources.insert(Resources::new(device, wgpu::TextureFormat::Rgba8Unorm));
        Self { resources }
    }

    pub fn update(
        &mut self,
        target: &mut Option<MapTexture>,
        render: &RenderState,
        ctx: &egui::Context,
        view_id: egui::Id,
        layers: &[(Arc<Preview>, f32)],
        layout: AtlasLayout,
    ) -> bool {
        if layers.is_empty() {
            *target = None;
            return false;
        }
        if target.as_ref().is_some_and(|t| {
            t.layout == layout
                && t.layers.len() == layers.len()
                && t.layers
                    .iter()
                    .zip(layers)
                    .all(|((a, x), (b, y))| Arc::ptr_eq(a, b) && x == y)
        }) {
            return true;
        }
        let mut geometries = Vec::with_capacity(layers.len());
        for (preview, opacity) in layers {
            let Some(geometry) = preview.projected_geometry_for_view(ProjectionKind::Mercator, ctx)
            else {
                return false;
            };
            geometries.push((geometry, *opacity));
        }
        let size = layout.size();
        let scale = (size[0].max(size[1]) as f64 / 2048.0).max(1.0);
        let size = size.map(|v| (f64::from(v) / scale).ceil().max(1.0) as u32);
        if target
            .as_ref()
            .is_none_or(|t| t.texture.width() != size[0] || t.texture.height() != size[1])
        {
            *target = Some(MapTexture::new(render, layout, size));
        }
        let target = target.as_mut().unwrap();
        // A virtual 2x display keeps outlines legible after the texture is warped.
        let frame = egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(size[0] as f32, size[1] as f32) / 2.0,
        );
        let bounds = layout.bounds();
        let south = bounds.min_lat.tan().asinh().to_degrees();
        let north = bounds.max_lat.tan().asinh().to_degrees();
        let shifts = if bounds.max_lon > 180.0 {
            &[0.0, -360.0][..]
        } else {
            &[0.0][..]
        };
        let mut callbacks = Vec::new();
        for (wrap, shift) in shifts.iter().enumerate() {
            let transform = PlotTransform::new(
                frame,
                egui_plot::PlotBounds::from_min_max(
                    [bounds.min_lon + shift, south],
                    [bounds.max_lon + shift, north],
                ),
                [false, false],
            );
            for (geometry, opacity) in &geometries {
                callbacks.push(PaintCallback::projected(
                    view_id.with(wrap),
                    Arc::clone(geometry),
                    *opacity,
                    &transform,
                ));
            }
        }
        let mut encoder = render.device.create_command_encoder(&Default::default());
        let screen = ScreenDescriptor {
            size_in_pixels: size,
            pixels_per_point: 2.0,
        };
        for callback in &callbacks {
            callback.prepare(
                &render.device,
                &render.queue,
                &screen,
                &mut encoder,
                &mut self.resources,
            );
        }
        let view = target.texture.create_view(&Default::default());
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("pushbroom_aoi_overlay"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            for callback in &callbacks {
                callback.paint(
                    egui::PaintCallbackInfo {
                        viewport: frame,
                        clip_rect: frame,
                        pixels_per_point: 2.0,
                        screen_size_px: size,
                    },
                    &mut pass,
                    &self.resources,
                );
            }
        }
        render.queue.submit([encoder.finish()]);
        target.layout = layout;
        target.layers = layers.to_vec();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_state() -> RenderState {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let renderer =
            egui_wgpu::Renderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, Default::default());
        RenderState {
            adapter,
            available_adapters: Vec::new(),
            device,
            queue,
            target_format: wgpu::TextureFormat::Rgba8Unorm,
            renderer: Arc::new(egui::mutex::RwLock::new(renderer)),
        }
    }

    fn rectangle(west: f64, east: f64, color: [u8; 3]) -> Arc<Preview> {
        let preview = Arc::new(Preview::from_geojson(serde_json::json!({
            "type":"Polygon", "coordinates":[[[west,-0.04],[east,-0.04],[east,0.04],[west,0.04],[west,-0.04]]]
        }), [(west+east)/2.0, 0.0], color).unwrap());
        preview.projected_geometry(ProjectionKind::Mercator);
        preview
    }

    fn pixels(render: &RenderState, target: &MapTexture) -> Vec<[u8; 4]> {
        let (width, height) = (target.texture.width(), target.texture.height());
        let stride = (width * 4).div_ceil(256) * 256;
        let buffer = render.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(stride) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = render.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            target.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        render.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        render
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let data = buffer.slice(..).get_mapped_range();
        let result = data
            .chunks(stride as usize)
            .flat_map(|row| {
                row[..width as usize * 4]
                    .chunks_exact(4)
                    .map(|p| p.try_into().unwrap())
            })
            .collect();
        drop(data);
        buffer.unmap();
        result
    }

    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn offscreen_aois_preserve_colors_transparency_and_cached_textures() {
        let render = render_state();
        let mut renderer = MapRenderer::new(&render.device);
        let ctx = egui::Context::default();
        let layers = vec![
            (rectangle(-0.06, -0.01, [255, 0, 0]), 1.0),
            (rectangle(0.01, 0.06, [0, 255, 0]), 1.0),
        ];
        let layout = AtlasLayout {
            x: 2047,
            y: 2047,
            z: 12,
            cols: 2,
            rows: 2,
        };
        let mut target = None;
        assert!(renderer.update(
            &mut target,
            &render,
            &ctx,
            egui::Id::new("test"),
            &layers,
            layout
        ));
        let texture = target.as_ref().unwrap();
        let id = texture.id;
        let image = pixels(&render, texture);
        let opaque_peak = image.iter().map(|p| p[3]).max().unwrap();
        assert!(
            image
                .iter()
                .filter(|p| p[0] > 100 && p[1] == 0 && p[3] > 100)
                .count()
                > 100
        );
        assert!(
            image
                .iter()
                .filter(|p| p[1] > 100 && p[0] == 0 && p[3] > 100)
                .count()
                > 100
        );
        assert_eq!(image[256 * 512 + 256], [0; 4]);
        assert!(renderer.update(
            &mut target,
            &render,
            &ctx,
            egui::Id::new("test"),
            &layers,
            layout
        ));
        assert_eq!(target.as_ref().unwrap().id, id);
        assert_eq!(
            renderer
                .resources
                .get::<Resources>()
                .unwrap()
                .geometry
                .len(),
            2
        );
        let faded = vec![(Arc::clone(&layers[0].0), 0.25)];
        assert!(renderer.update(
            &mut target,
            &render,
            &ctx,
            egui::Id::new("test"),
            &faded,
            layout
        ));
        assert_eq!(target.as_ref().unwrap().id, id);
        let image = pixels(&render, target.as_ref().unwrap());
        assert!(image.iter().any(|p| p[0] > 20));
        assert!(image.iter().all(|p| p[1] == 0));
        assert!(image.iter().map(|p| p[3]).max().unwrap() < opaque_peak);
        assert!(!renderer.update(
            &mut target,
            &render,
            &ctx,
            egui::Id::new("test"),
            &[],
            layout
        ));
        assert!(target.is_none());
        assert!(render.renderer.read().texture(&id).is_none());
    }

    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn offscreen_aois_cover_both_sides_of_the_dateline() {
        let render = render_state();
        let mut renderer = MapRenderer::new(&render.device);
        let layers = vec![
            (rectangle(179.94, 179.98, [255, 0, 0]), 1.0),
            (rectangle(-179.98, -179.94, [0, 255, 0]), 1.0),
        ];
        let layout = AtlasLayout {
            x: 4095,
            y: 2047,
            z: 12,
            cols: 2,
            rows: 2,
        };
        let mut target = None;
        assert!(renderer.update(
            &mut target,
            &render,
            &egui::Context::default(),
            egui::Id::new("dateline"),
            &layers,
            layout
        ));
        let image = pixels(&render, target.as_ref().unwrap());
        let left: Vec<_> = image.chunks(512).flat_map(|row| &row[..256]).collect();
        let right: Vec<_> = image.chunks(512).flat_map(|row| &row[256..]).collect();
        assert!(left.iter().any(|p| p[0] > 100 && p[1] == 0));
        assert!(right.iter().any(|p| p[1] > 100 && p[0] == 0));
        assert!(!left.iter().any(|p| p[1] > 0));
        assert!(!right.iter().any(|p| p[0] > 0));
    }
}
