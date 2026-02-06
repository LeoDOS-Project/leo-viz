//! Image texture loading, decoding, and processing.
//!
//! Handles loading and sampling planet textures (JPEG/PNG), CPU-based
//! sphere rendering, asset path resolution, and WASM texture fetching.

use egui::Color32;
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;
use std::sync::Arc;

#[allow(dead_code)]
pub enum TextureLoadState {
    Loading,
    Loaded(Arc<EarthTexture>),
    Failed(String),
}

#[derive(Clone)]
pub struct EarthTexture {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[u8; 3]>,
}

impl EarthTexture {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_path(path: &std::path::Path) -> Self {
        let bytes = std::fs::read(path)
            .expect("Failed to read earth texture");
        Self::from_bytes(&bytes).expect("Failed to load Earth texture")
    }

    #[cfg(target_arch = "wasm32")]
    pub fn default_placeholder() -> Self {
        Self { width: 2, height: 1, pixels: vec![[30, 60, 120], [30, 60, 120]] }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        use std::io::Cursor;
        let cursor = Cursor::new(bytes);
        let mut reader = image::ImageReader::new(cursor)
            .with_guessed_format()
            .map_err(|e| format!("Failed to guess format: {}", e))?;
        reader.no_limits();
        let img = reader.decode()
            .map_err(|e| format!("Failed to decode image: {}", e))?
            .to_rgb8();
        let width = img.width();
        let height = img.height();
        let pixels: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();
        Ok(Self { width, height, pixels })
    }

    pub fn downscale(&self, factor: u32) -> Self {
        if factor <= 1 {
            return self.clone();
        }
        let new_width = self.width / factor;
        let new_height = self.height / factor;
        let mut pixels = Vec::with_capacity((new_width * new_height) as usize);

        for y in 0..new_height {
            for x in 0..new_width {
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                for dy in 0..factor {
                    for dx in 0..factor {
                        let sx = x * factor + dx;
                        let sy = y * factor + dy;
                        let idx = (sy * self.width + sx) as usize;
                        let [r, g, b] = self.pixels[idx];
                        r_sum += r as u32;
                        g_sum += g as u32;
                        b_sum += b as u32;
                    }
                }
                let count = factor * factor;
                pixels.push([
                    (r_sum / count) as u8,
                    (g_sum / count) as u8,
                    (b_sum / count) as u8,
                ]);
            }
        }
        Self { width: new_width, height: new_height, pixels }
    }

    pub fn sample(&self, u: f64, v: f64) -> [u8; 3] {
        let x = ((u * self.width as f64) as u32).min(self.width - 1);
        let y = ((v * self.height as f64) as u32).min(self.height - 1);
        self.pixels[(y * self.width + x) as usize]
    }

    pub fn render_sphere(&self, size: usize, rot: &Matrix3<f64>, flattening: f64) -> egui::ColorImage {
        let mut pixels = vec![Color32::TRANSPARENT; size * size];
        let center = size as f64 / 2.0;
        let radius = center;
        let inv_rot = rot.transpose();
        let polar_scale = 1.0 - flattening;

        for py in 0..size {
            for px in 0..size {
                let dx = px as f64 - center;
                let dy = py as f64 - center;
                let dy_scaled = dy / polar_scale;
                let dist_sq = dx * dx + dy_scaled * dy_scaled;

                if dist_sq < radius * radius {
                    let z = (radius * radius - dist_sq).sqrt();
                    let x = dx / radius;
                    let y = -dy_scaled / radius;
                    let z = z / radius;

                    let v = inv_rot * Vector3::new(x, y, z);

                    let lat = v.y.asin();
                    let lon = (-v.z).atan2(v.x);

                    let u = (lon + PI) / (2.0 * PI);
                    let vt = (PI / 2.0 - lat) / PI;

                    let [r, g, b] = self.sample(u, vt);

                    let shade = (0.3 + 0.7 * z.max(0.0)) as f32;
                    let r = (r as f32 * shade) as u8;
                    let g = (g as f32 * shade) as u8;
                    let b = (b as f32 * shade) as u8;

                    pixels[py * size + px] = Color32::from_rgb(r, g, b);
                }
            }
        }

        egui::ColorImage {
            size: [size, size],
            pixels,
            source_size: egui::Vec2::ZERO,
        }
    }
}

pub struct RingTexture {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[u8; 4]>,
}

impl RingTexture {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        use std::io::Cursor;
        let cursor = Cursor::new(bytes);
        let mut reader = image::ImageReader::new(cursor)
            .with_guessed_format()
            .map_err(|e| format!("Failed to guess format: {}", e))?;
        reader.no_limits();
        let img = reader.decode()
            .map_err(|e| format!("Failed to decode image: {}", e))?
            .to_rgba8();
        let width = img.width();
        let height = img.height();
        let pixels: Vec<[u8; 4]> = img.pixels().map(|p| p.0).collect();
        Ok(Self { width, height, pixels })
    }
}

pub(crate) fn asset_path(relative: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn decode_jpeg_pixels(bytes: &[u8]) -> Option<(Vec<[u8; 3]>, u32, u32)> {
    use std::io::Cursor;
    let img = image::load(Cursor::new(bytes), image::ImageFormat::Jpeg).ok()?;
    let rgb = img.to_rgb8();
    let w = rgb.width();
    let h = rgb.height();
    let pixels: Vec<[u8; 3]> = rgb.pixels().map(|p| p.0).collect();
    Some((pixels, w, h))
}

pub(crate) fn load_earth_texture() -> EarthTexture {
    #[cfg(not(target_arch = "wasm32"))]
    {
        EarthTexture::load_from_path(&asset_path("textures/earth/earth_8k.jpg"))
    }
    #[cfg(target_arch = "wasm32")]
    {
        EarthTexture::default_placeholder()
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    pub(crate) static TEXTURE_RESULT: std::cell::RefCell<Vec<((crate::celestial::CelestialBody, crate::celestial::Skin, crate::celestial::TextureResolution), Result<EarthTexture, String>)>> = std::cell::RefCell::new(Vec::new());
    pub(crate) static STAR_TEXTURE_RESULT: std::cell::RefCell<Option<Result<EarthTexture, String>>> = std::cell::RefCell::new(None);
    pub(crate) static MILKY_WAY_TEXTURE_RESULT: std::cell::RefCell<Option<Result<EarthTexture, String>>> = std::cell::RefCell::new(None);
    pub(crate) static NIGHT_TEXTURE_RESULT: std::cell::RefCell<Option<Result<EarthTexture, String>>> = std::cell::RefCell::new(None);
    pub(crate) static CLOUD_TEXTURE_RESULT: std::cell::RefCell<Option<(crate::celestial::TextureResolution, Result<EarthTexture, String>)>> = std::cell::RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn fetch_texture(url: &str) -> Result<EarthTexture, String> {
    use wasm_bindgen::JsCast as _;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts)
        .map_err(|e| format!("Failed to create request: {:?}", e))?;

    let window = web_sys::window().ok_or("No window")?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| format!("Fetch failed: {:?}", e))?;

    let resp: Response = resp_value.dyn_into()
        .map_err(|_| "Response is not a Response")?;

    if !resp.ok() {
        return Err(format!("HTTP error: {}", resp.status()));
    }

    let array_buffer = wasm_bindgen_futures::JsFuture::from(
        resp.array_buffer().map_err(|e| format!("Failed to get array buffer: {:?}", e))?
    )
    .await
    .map_err(|e| format!("Failed to read response: {:?}", e))?;

    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    let bytes: Vec<u8> = uint8_array.to_vec();

    EarthTexture::from_bytes(&bytes)
}
