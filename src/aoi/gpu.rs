//! Persistent body-fixed geometry; rotation, clipping, and detail selection run on the GPU.

use egui_plot::{PlotPoint, PlotTransform};
use egui_wgpu::{wgpu, CallbackResources, CallbackTrait, ScreenDescriptor};
use nalgebra::Matrix3;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Weak,
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Segment {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub anchor: [f32; 3],
    pub radius: f32,
    pub color: [u8; 4],
    pub scales: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Dot {
    pub anchor: [f32; 3],
    pub radius: f32,
    pub color: [u8; 4],
}

pub(super) struct Geometry {
    id: usize,
    segments: Vec<Segment>,
    dots: Vec<Dot>,
}

impl Geometry {
    pub fn new(segments: Vec<Segment>, dots: Vec<Dot>) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            segments,
            dots,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    rotation: [[f32; 4]; 3],
    screen: [f32; 4],
    viewport: [f32; 4],
    params: [f32; 4],
}

struct UploadedGeometry {
    owner: Weak<Geometry>,
    segments: Vec<(wgpu::Buffer, u32)>,
    dots: wgpu::Buffer,
    dot_count: u32,
}

struct View {
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct Resources {
    lines: wgpu::RenderPipeline,
    dots: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    geometry: HashMap<usize, UploadedGeometry>,
    views: HashMap<(egui::Id, usize), View>,
}

impl Resources {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aoi_uniform_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as u64),
                },
                count: None,
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aoi_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aoi_pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline =
            |entry: &'static str, stride: usize, attributes: &[wgpu::VertexAttribute]| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: stride as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes,
                        }],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fragment"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
            };
        let lines = pipeline(
            "line_vertex",
            std::mem::size_of::<Segment>(),
            &wgpu::vertex_attr_array![
                0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Unorm8x4, 4 => Float32x2,
            ],
        );
        let dots = pipeline(
            "dot_vertex",
            std::mem::size_of::<Dot>(),
            &wgpu::vertex_attr_array![
                0 => Float32x3, 1 => Float32, 2 => Unorm8x4,
            ],
        );
        Self {
            lines,
            dots,
            layout,
            format,
            geometry: HashMap::new(),
            views: HashMap::new(),
        }
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        callback: &PaintCallback,
        ppp: f32,
    ) {
        self.geometry
            .retain(|_, entry| entry.owner.strong_count() > 0);
        self.views
            .retain(|(_, id), _| self.geometry.contains_key(id));
        self.geometry
            .entry(callback.geometry.id)
            .or_insert_with(|| {
                let upload = |label, bytes: &[u8]| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: if bytes.is_empty() { &[0; 4] } else { bytes },
                        usage: wgpu::BufferUsages::VERTEX,
                    })
                };
                UploadedGeometry {
                    owner: Arc::downgrade(&callback.geometry),
                    segments: callback
                        .geometry
                        .segments
                        .chunks(
                            (device.limits().max_buffer_size
                                / std::mem::size_of::<Segment>() as u64)
                                .min(u64::from(u32::MAX))
                                .max(1) as usize,
                        )
                        .map(|chunk| {
                            (
                                upload("aoi_segments", bytemuck::cast_slice(chunk)),
                                chunk.len() as u32,
                            )
                        })
                        .collect(),
                    dots: upload("aoi_dots", bytemuck::cast_slice(&callback.geometry.dots)),
                    dot_count: callback.geometry.dots.len() as u32,
                }
            });
        let view = self
            .views
            .entry((callback.view_id, callback.geometry.id))
            .or_insert_with(|| {
                let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("aoi_view_uniforms"),
                    size: std::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("aoi_view_bind_group"),
                    layout: &self.layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniforms.as_entire_binding(),
                    }],
                });
                View {
                    uniforms,
                    bind_group,
                }
            });
        let mut uniforms = callback.uniforms;
        uniforms.viewport[2] = ppp;
        uniforms.params[2] = if self.format.is_srgb() { 1.0 } else { 0.0 };
        queue.write_buffer(&view.uniforms, 0, bytemuck::bytes_of(&uniforms));
    }
}

pub(super) struct PaintCallback {
    view_id: egui::Id,
    geometry: Arc<Geometry>,
    uniforms: Uniforms,
}

impl PaintCallback {
    pub fn new(
        view_id: egui::Id,
        geometry: Arc<Geometry>,
        rotation: &Matrix3<f64>,
        radius: f64,
        polar_scale: f64,
        opacity: f32,
        transform: &PlotTransform,
    ) -> Self {
        let origin =
            transform.position_from_point(&PlotPoint::new(0.0, 0.0)) - transform.frame().min;
        let scale = transform
            .dpos_dvalue_x()
            .abs()
            .max(transform.dpos_dvalue_y().abs())
            * radius;
        Self {
            view_id,
            geometry,
            uniforms: Uniforms {
                rotation: std::array::from_fn(|row| {
                    [
                        rotation[(row, 0)] as f32,
                        rotation[(row, 1)] as f32,
                        rotation[(row, 2)] as f32,
                        0.0,
                    ]
                }),
                screen: [
                    origin.x,
                    origin.y,
                    (transform.dpos_dvalue_x() * radius) as f32,
                    (transform.dpos_dvalue_y() * radius) as f32,
                ],
                viewport: [
                    transform.frame().width(),
                    transform.frame().height(),
                    1.0,
                    opacity,
                ],
                params: [polar_scale as f32, scale as f32, 0.0, 0.0],
            },
        }
    }
}

impl CallbackTrait for PaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if resources.get::<Resources>().is_none() {
            let format = resources
                .get::<crate::renderer::GpuResources>()
                .unwrap()
                ._target_format;
            resources.insert(Resources::new(device, format));
        }
        resources.get_mut::<Resources>().unwrap().prepare(
            device,
            queue,
            self,
            screen.pixels_per_point,
        );
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let resources = resources.get::<Resources>().unwrap();
        let Some(geometry) = resources.geometry.get(&self.geometry.id) else {
            return;
        };
        let Some(view) = resources.views.get(&(self.view_id, self.geometry.id)) else {
            return;
        };
        pass.set_bind_group(0, &view.bind_group, &[]);
        pass.set_pipeline(&resources.lines);
        for (buffer, count) in &geometry.segments {
            pass.set_vertex_buffer(0, buffer.slice(..));
            pass.draw(0..6, 0..*count);
        }
        pass.set_pipeline(&resources.dots);
        pass.set_vertex_buffer(0, geometry.dots.slice(..));
        pass.draw(0..6, 0..geometry.dot_count);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
