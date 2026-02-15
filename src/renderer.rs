//! GPU-based sphere rendering with shaders.
//!
//! Handles OpenGL/WebGL rendering of planetary bodies with textures,
//! atmosphere effects, day/night cycles, clouds, rings, and starfields.

use eframe::glow;
use glow::HasContext as _;
use nalgebra::Matrix3;
use std::collections::HashMap;

use crate::celestial::{CelestialBody, Skin, TextureResolution};
use crate::texture::{EarthTexture, RingTexture};
use crate::tile::DetailTexture;

pub struct SphereRenderer {
    program: glow::Program,
    vertex_array: glow::VertexArray,
    textures: HashMap<(CelestialBody, Skin, TextureResolution), glow::Texture>,
    cloud_textures: HashMap<TextureResolution, glow::Texture>,
    night_texture: Option<glow::Texture>,
    star_texture: Option<glow::Texture>,
    milky_way_texture: Option<glow::Texture>,
    ring_textures: HashMap<CelestialBody, glow::Texture>,
}

impl SphereRenderer {
    pub fn new(gl: &glow::Context) -> Self {
        let shader_version = if cfg!(target_arch = "wasm32") {
            "#version 300 es"
        } else {
            "#version 330"
        };

        unsafe {
            let program = gl.create_program().expect("Cannot create program");

            let vertex_shader_source = r#"
                const vec2 verts[4] = vec2[4](
                    vec2(-1.0, -1.0),
                    vec2( 1.0, -1.0),
                    vec2(-1.0,  1.0),
                    vec2( 1.0,  1.0)
                );
                out vec2 v_uv;
                void main() {
                    v_uv = verts[gl_VertexID] * 0.5 + 0.5;
                    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
                }
            "#;

            let fragment_shader_source = r#"
                precision highp float;
                in vec2 v_uv;
                out vec4 out_color;

                uniform sampler2D u_texture;
                uniform sampler2D u_clouds;
                uniform sampler2D u_night;
                uniform sampler2D u_detail;
                uniform sampler2D u_stars;
                uniform mat3 u_inv_rotation;
                uniform float u_flattening;
                uniform float u_aspect;
                uniform float u_scale;
                uniform float u_atmosphere;
                uniform float u_show_clouds;
                uniform float u_show_day_night;
                uniform vec3 u_sun_dir;
                uniform vec4 u_detail_bounds;
                uniform float u_use_detail;
                uniform float u_show_stars;
                uniform vec3 u_bg_color;
                uniform sampler2D u_ring_texture;
                uniform float u_has_rings;
                uniform float u_ring_inner;
                uniform float u_ring_outer;
                uniform float u_adams_arc;
                uniform float u_epsilon_wobble;
                uniform float u_transparent_bg;
                uniform vec2 u_uv_scale;

                const float PI = 3.14159265359;
                const vec3 ATMO_COLOR = vec3(0.4, 0.7, 1.0);
                const float ATMO_THICKNESS = 0.06;

                void main() {
                    vec2 centered = (v_uv * u_uv_scale - 0.5) * 2.0;
                    centered.x *= max(u_aspect, 1.0);
                    centered.y *= max(1.0 / u_aspect, 1.0);
                    centered /= u_scale;

                    float b = 1.0 - u_flattening;
                    float b2 = b * b;

                    vec3 O = u_inv_rotation * vec3(centered.x, centered.y, 0.0);
                    vec3 D = u_inv_rotation * vec3(0.0, 0.0, -1.0);

                    float A = D.x*D.x + D.y*D.y/b2 + D.z*D.z;
                    float B = 2.0 * (O.x*D.x + O.y*D.y/b2 + O.z*D.z);
                    float C = O.x*O.x + O.y*O.y/b2 + O.z*O.z - 1.0;

                    float discriminant = B*B - 4.0*A*C;

                    float screen_dist = length(centered);
                    float atmo_outer = 1.0 + ATMO_THICKNESS * u_atmosphere;

                    float lat_ortho = 0.0;
                    float lon_ortho = 0.0;
                    vec3 normal_ortho = vec3(0.0, 0.0, 1.0);
                    bool ortho_hit = discriminant >= 0.0;

                    if (ortho_hit) {
                        float t = (-B - sqrt(discriminant)) / (2.0 * A);
                        vec3 world_pt = O + t * D;
                        lat_ortho = asin(clamp(world_pt.y / b, -1.0, 1.0));
                        lon_ortho = atan(-world_pt.z, world_pt.x);
                        normal_ortho = normalize(vec3(world_pt.x, world_pt.y / b2, world_pt.z));
                    }

                    float lat, lon;
                    vec3 normal;
                    float alpha;

                    float t_sphere = ortho_hit ? (-B - sqrt(discriminant)) / (2.0 * A) : 1e10;

                    float ring_alpha = 0.0;
                    vec3 ring_color = vec3(0.0);
                    float t_ring = 1e10;
                    if (u_has_rings > 0.5 && abs(D.y) > 0.0001) {
                        float t_disc = -O.y / D.y;
                        vec3 rh = O + t_disc * D;
                        float r = length(vec2(rh.x, rh.z));
                        if (u_epsilon_wobble > 0.5) {
                            float theta = atan(rh.z, rh.x);
                            float eps_center = 2.017;
                            float eps_zone = 0.06;
                            float prox = 1.0 - smoothstep(0.0, eps_zone, abs(r - eps_center));
                            if (prox > 0.0) {
                                float radial_shift = eps_center * 0.025 * cos(theta);
                                r -= radial_shift * prox;
                                float width_scale = 1.0 + 0.7 * cos(theta);
                                float dr = r - eps_center;
                                r = eps_center + dr / mix(1.0, width_scale, prox);
                            }
                        }
                        if (r >= u_ring_inner && r <= u_ring_outer) {
                            float ru = (r - u_ring_inner) / (u_ring_outer - u_ring_inner);
                            vec4 rs = texture(u_ring_texture, vec2(ru, 0.5));
                            ring_color = rs.rgb;
                            ring_alpha = rs.a;
                            if (u_adams_arc > 0.5 && ru > 0.82) {
                                float ang = atan(rh.z, rh.x);
                                float deg = ang * 180.0 / PI;
                                if (deg < 0.0) deg += 360.0;
                                float arc = 0.0;
                                if (deg > 237.0 && deg < 257.0) arc = smoothstep(237.0, 240.0, deg) * (1.0 - smoothstep(254.0, 257.0, deg));
                                if (deg > 258.0 && deg < 265.0) arc = max(arc, smoothstep(258.0, 260.0, deg) * (1.0 - smoothstep(263.0, 265.0, deg)));
                                if (deg > 266.0 && deg < 273.0) arc = max(arc, smoothstep(266.0, 268.0, deg) * (1.0 - smoothstep(271.0, 273.0, deg)));
                                if (deg > 274.0 && deg < 290.0) arc = max(arc, smoothstep(274.0, 277.0, deg) * (1.0 - smoothstep(287.0, 290.0, deg)));
                                if (deg > 292.0 && deg < 320.0) arc = max(arc, smoothstep(292.0, 296.0, deg) * (1.0 - smoothstep(316.0, 320.0, deg)));
                                ring_alpha *= mix(0.15, 1.0, arc);
                            }
                            t_ring = t_disc;
                        }
                    }

                    bool ring_in_front = ring_alpha > 0.01 && t_ring < t_sphere;

                    if (!ortho_hit && ring_alpha < 0.01) {
                        vec3 bg = vec3(0.0);
                        float bg_alpha = 0.0;

                        if (u_show_stars > 0.5) {
                            float vp_aspect = u_aspect * u_uv_scale.x / u_uv_scale.y;
                            vec2 sp = (v_uv - 0.5) * 2.0;
                            sp.x *= vp_aspect;
                            vec3 dir = u_inv_rotation * normalize(vec3(sp, -2.0));
                            float slat = asin(clamp(dir.y, -1.0, 1.0));
                            float slon = atan(-dir.z, dir.x);
                            float su = (slon + PI) / (2.0 * PI);
                            float sv = (PI / 2.0 - slat) / PI;
                            bg = texture(u_stars, vec2(su, sv)).rgb;
                            bg_alpha = 1.0;
                        }

                        if (u_atmosphere > 0.0 && screen_dist < atmo_outer) {
                            float C_atmo = O.x*O.x + O.y*O.y + O.z*O.z - 1.0;
                            float disc_atmo = B*B - 4.0*A*C_atmo;
                            if (disc_atmo >= 0.0) {
                                float atmo_depth = (screen_dist - 1.0) / (ATMO_THICKNESS * u_atmosphere);
                                atmo_depth = clamp(atmo_depth, 0.0, 1.0);
                                float atmo_falloff = 1.0 - atmo_depth;
                                atmo_falloff = pow(atmo_falloff, 2.0);
                                float glow = atmo_falloff * 0.8;
                                bg = bg * (1.0 - glow) + ATMO_COLOR * glow;
                                bg_alpha = max(bg_alpha, glow);
                            }
                        }

                        if (u_transparent_bg > 0.5) { discard; }
                        out_color = vec4(mix(u_bg_color, bg, bg_alpha), 1.0);
                        return;
                    }

                    if (!ortho_hit && ring_alpha >= 0.01) {
                        vec3 bg = vec3(0.0);
                        float bg_alpha = 0.0;
                        if (u_show_stars > 0.5) {
                            float vp_aspect = u_aspect * u_uv_scale.x / u_uv_scale.y;
                            vec2 sp = (v_uv - 0.5) * 2.0;
                            sp.x *= vp_aspect;
                            vec3 dir = u_inv_rotation * normalize(vec3(sp, -2.0));
                            float slat = asin(clamp(dir.y, -1.0, 1.0));
                            float slon = atan(-dir.z, dir.x);
                            float su = (slon + PI) / (2.0 * PI);
                            float sv = (PI / 2.0 - slat) / PI;
                            bg = texture(u_stars, vec2(su, sv)).rgb;
                            bg_alpha = 1.0;
                        }
                        if (u_transparent_bg > 0.5) {
                            out_color = vec4(ring_color, ring_alpha);
                            return;
                        }
                        vec3 base = mix(u_bg_color, bg, bg_alpha);
                        vec3 final_color = mix(base, ring_color, ring_alpha);
                        out_color = vec4(final_color, 1.0);
                        return;
                    }
                    lat = lat_ortho;
                    lon = lon_ortho;
                    normal = normal_ortho;
                    alpha = 1.0;

                    float tex_u = (lon + PI) / (2.0 * PI);
                    float tex_v = (PI / 2.0 - lat) / PI;

                    vec3 day_color;
                    if (u_use_detail > 0.5) {
                        float lon_deg = lon * 180.0 / PI;
                        if (lon_deg < u_detail_bounds.x) lon_deg += 360.0;
                        float du = (lon_deg - u_detail_bounds.x) / (u_detail_bounds.y - u_detail_bounds.x);
                        float dv = (u_detail_bounds.w - lat) / (u_detail_bounds.w - u_detail_bounds.z);
                        if (du >= 0.0 && du <= 1.0 && dv >= 0.0 && dv <= 1.0) {
                            day_color = texture(u_detail, vec2(du, dv)).rgb;
                        } else {
                            day_color = texture(u_texture, vec2(tex_u, tex_v)).rgb;
                        }
                    } else {
                        day_color = texture(u_texture, vec2(tex_u, tex_v)).rgb;
                    }

                    if (u_show_clouds > 0.5 && u_use_detail < 0.5) {
                        float cloud = texture(u_clouds, vec2(tex_u, tex_v)).r;
                        day_color = mix(day_color, vec3(1.0), cloud);
                    }

                    vec3 color;
                    float sun_dot = dot(normal, u_sun_dir);
                    if (u_show_day_night > 0.5) {
                        float day_factor = smoothstep(-0.1, 0.1, sun_dot);
                        float shade = 0.2 + 0.8 * max(sun_dot, 0.0);
                        vec3 lit_day = day_color * shade;
                        vec3 night_lights = texture(u_night, vec2(tex_u, tex_v)).rgb;
                        color = mix(night_lights, lit_day, day_factor);
                    } else {
                        float shade = 0.3 + 0.7 * max(dot(normal, -D), 0.0);
                        color = day_color * shade;
                    }

                    if (u_atmosphere > 0.0) {
                        float fresnel = 1.0 - max(dot(normal, -D), 0.0);
                        fresnel = pow(fresnel, 3.0);
                        float rim = fresnel * 0.6 * u_atmosphere;
                        float atmo_sun = u_show_day_night > 0.5 ? max(sun_dot + 0.3, 0.0) : 1.0;
                        color = mix(color, ATMO_COLOR * atmo_sun, rim);
                    }

                    if (ring_in_front) {
                        color = mix(color, ring_color, ring_alpha);
                    }

                    if (u_transparent_bg > 0.5) {
                        out_color = vec4(color, 1.0);
                    } else {
                        out_color = vec4(mix(u_bg_color, color, alpha), 1.0);
                    }
                }
            "#;

            let shader_sources = [
                (glow::VERTEX_SHADER, vertex_shader_source),
                (glow::FRAGMENT_SHADER, fragment_shader_source),
            ];

            let shaders: Vec<_> = shader_sources
                .iter()
                .map(|(shader_type, shader_source)| {
                    let shader = gl.create_shader(*shader_type).expect("Cannot create shader");
                    gl.shader_source(shader, &format!("{shader_version}\n{shader_source}"));
                    gl.compile_shader(shader);
                    assert!(
                        gl.get_shader_compile_status(shader),
                        "Failed to compile shader: {}",
                        gl.get_shader_info_log(shader)
                    );
                    gl.attach_shader(program, shader);
                    shader
                })
                .collect();

            gl.link_program(program);
            assert!(
                gl.get_program_link_status(program),
                "Failed to link program: {}",
                gl.get_program_info_log(program)
            );

            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }

            let vertex_array = gl.create_vertex_array().expect("Cannot create vertex array");

            Self {
                program,
                vertex_array,
                textures: HashMap::new(),
                cloud_textures: HashMap::new(),
                night_texture: None,
                star_texture: None,
                milky_way_texture: None,
                ring_textures: HashMap::new(),
            }
        }
    }

    pub fn upload_night_texture(&mut self, gl: &glow::Context, night_tex: &EarthTexture) {
        unsafe {
            if self.night_texture.is_some() {
                return;
            }

            let texture = gl.create_texture().expect("Cannot create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            let pixels: Vec<u8> = night_tex.pixels.iter()
                .flat_map(|&[r, g, b]| [r, g, b])
                .collect();

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB as i32,
                night_tex.width as i32,
                night_tex.height as i32,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            self.night_texture = Some(texture);
        }
    }

    pub fn upload_star_texture(&mut self, gl: &glow::Context, tex: &EarthTexture) {
        unsafe {
            if self.star_texture.is_some() {
                return;
            }
            let texture = gl.create_texture().expect("Cannot create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            let pixels: Vec<u8> = tex.pixels.iter().flat_map(|&[r, g, b]| [r, g, b]).collect();
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGB as i32,
                tex.width as i32, tex.height as i32, 0,
                glow::RGB, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            self.star_texture = Some(texture);
        }
    }

    pub fn upload_milky_way_texture(&mut self, gl: &glow::Context, tex: &EarthTexture) {
        unsafe {
            if self.milky_way_texture.is_some() {
                return;
            }
            let texture = gl.create_texture().expect("Cannot create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            let pixels: Vec<u8> = tex.pixels.iter().flat_map(|&[r, g, b]| [r, g, b]).collect();
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGB as i32,
                tex.width as i32, tex.height as i32, 0,
                glow::RGB, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            self.milky_way_texture = Some(texture);
        }
    }

    pub fn upload_ring_texture(&mut self, gl: &glow::Context, body: CelestialBody, tex: &RingTexture) {
        unsafe {
            if self.ring_textures.contains_key(&body) {
                return;
            }
            let texture = gl.create_texture().expect("Cannot create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            let pixels: Vec<u8> = tex.pixels.iter().flat_map(|&[r, g, b, a]| [r, g, b, a]).collect();
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA as i32,
                tex.width as i32, tex.height as i32, 0,
                glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            self.ring_textures.insert(body, texture);
        }
    }

    pub fn upload_texture(&mut self, gl: &glow::Context, key: (CelestialBody, Skin, TextureResolution), earth_tex: &EarthTexture) {
        unsafe {
            if self.textures.contains_key(&key) {
                return;
            }

            let texture = gl.create_texture().expect("Cannot create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            let pixels: Vec<u8> = earth_tex.pixels.iter()
                .flat_map(|&[r, g, b]| [r, g, b])
                .collect();

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB as i32,
                earth_tex.width as i32,
                earth_tex.height as i32,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );

            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

            self.textures.insert(key, texture);
        }
    }

    pub fn evict_unused_textures(&mut self, gl: &glow::Context, keep: &[(CelestialBody, Skin, TextureResolution)]) {
        let to_remove: Vec<_> = self.textures.keys()
            .filter(|k| !keep.contains(k))
            .copied()
            .collect();
        for key in to_remove {
            if let Some(tex) = self.textures.remove(&key) {
                unsafe { gl.delete_texture(tex); }
            }
        }
        let keep_bodies: std::collections::HashSet<CelestialBody> = keep.iter().map(|k| k.0).collect();
        let rings_to_remove: Vec<_> = self.ring_textures.keys()
            .filter(|b| !keep_bodies.contains(b))
            .copied()
            .collect();
        for body in rings_to_remove {
            if let Some(tex) = self.ring_textures.remove(&body) {
                unsafe { gl.delete_texture(tex); }
            }
        }
    }

    pub fn upload_cloud_texture(&mut self, gl: &glow::Context, res: TextureResolution, cloud_tex: &EarthTexture) {
        unsafe {
            if self.cloud_textures.contains_key(&res) {
                return;
            }

            let texture = gl.create_texture().expect("Cannot create cloud texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            let pixels: Vec<u8> = cloud_tex.pixels.iter()
                .flat_map(|&[r, g, b]| [r, g, b])
                .collect();

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB as i32,
                cloud_tex.width as i32,
                cloud_tex.height as i32,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );

            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

            self.cloud_textures.insert(res, texture);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        &self,
        gl: &glow::Context,
        key: (CelestialBody, Skin, TextureResolution),
        inv_rotation: &Matrix3<f64>,
        flattening: f64,
        aspect: f32,
        scale: f32,
        atmosphere: f32,
        show_clouds: bool,
        show_day_night: bool,
        sun_dir: [f32; 3],
        detail_texture: Option<&DetailTexture>,
        show_stars: bool,
        show_milky_way: bool,
        bg_color: [f32; 3],
        uv_scale: [f32; 2],
    ) {
        let Some(texture) = self.textures.get(&key) else { return };

        unsafe {
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vertex_array));

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_texture").as_ref(), 0);

            gl.active_texture(glow::TEXTURE1);
            let cloud_tex = self.cloud_textures.get(&key.2);
            if let Some(ct) = cloud_tex {
                gl.bind_texture(glow::TEXTURE_2D, Some(*ct));
            } else {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            }
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_clouds").as_ref(), 1);

            gl.active_texture(glow::TEXTURE2);
            if let Some(nt) = self.night_texture {
                gl.bind_texture(glow::TEXTURE_2D, Some(nt));
            } else {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            }
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_night").as_ref(), 2);

            gl.active_texture(glow::TEXTURE3);
            let use_detail = if let Some(dt) = detail_texture {
                if let Some(gl_tex) = dt.gl_texture {
                    gl.bind_texture(glow::TEXTURE_2D, Some(gl_tex));
                    gl.uniform_4_f32(
                        gl.get_uniform_location(self.program, "u_detail_bounds").as_ref(),
                        dt.bounds.min_lon as f32,
                        dt.bounds.max_lon as f32,
                        dt.bounds.min_lat as f32,
                        dt.bounds.max_lat as f32,
                    );
                    true
                } else {
                    gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
                    false
                }
            } else {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
                false
            };
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_detail").as_ref(), 3);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_use_detail").as_ref(), if use_detail { 1.0 } else { 0.0 });

            gl.active_texture(glow::TEXTURE4);
            let star_tex = if show_milky_way { self.milky_way_texture } else { self.star_texture };
            if let Some(st) = star_tex {
                gl.bind_texture(glow::TEXTURE_2D, Some(st));
            } else {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            }
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_stars").as_ref(), 4);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_show_stars").as_ref(), if show_stars && star_tex.is_some() { 1.0 } else { 0.0 });

            gl.active_texture(glow::TEXTURE5);
            let ring_params = key.0.ring_params();
            let has_rings = if let Some(rt) = self.ring_textures.get(&key.0) {
                gl.bind_texture(glow::TEXTURE_2D, Some(*rt));
                true
            } else {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
                false
            };
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_ring_texture").as_ref(), 5);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_has_rings").as_ref(), if has_rings { 1.0 } else { 0.0 });
            let (ring_inner, ring_outer) = ring_params.map(|(_, i, o)| (i, o)).unwrap_or((0.0, 0.0));
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_ring_inner").as_ref(), ring_inner);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_ring_outer").as_ref(), ring_outer);
            let adams_arc = if has_rings && key.0 == CelestialBody::Neptune { 1.0f32 } else { 0.0f32 };
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_adams_arc").as_ref(), adams_arc);
            let epsilon_wobble = if has_rings && key.0 == CelestialBody::Uranus { 1.0f32 } else { 0.0f32 };
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_epsilon_wobble").as_ref(), epsilon_wobble);

            let rot_data: [f32; 9] = [
                inv_rotation[(0, 0)] as f32, inv_rotation[(1, 0)] as f32, inv_rotation[(2, 0)] as f32,
                inv_rotation[(0, 1)] as f32, inv_rotation[(1, 1)] as f32, inv_rotation[(2, 1)] as f32,
                inv_rotation[(0, 2)] as f32, inv_rotation[(1, 2)] as f32, inv_rotation[(2, 2)] as f32,
            ];
            gl.uniform_matrix_3_f32_slice(
                gl.get_uniform_location(self.program, "u_inv_rotation").as_ref(),
                false,
                &rot_data,
            );

            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_flattening").as_ref(), flattening as f32);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_aspect").as_ref(), aspect);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_scale").as_ref(), scale);
            gl.uniform_2_f32(gl.get_uniform_location(self.program, "u_uv_scale").as_ref(), uv_scale[0], uv_scale[1]);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_atmosphere").as_ref(), atmosphere);
            let clouds_enabled = show_clouds && cloud_tex.is_some() && key.0 == CelestialBody::Earth;
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_show_clouds").as_ref(), if clouds_enabled { 1.0 } else { 0.0 });

            let day_night_enabled = show_day_night && self.night_texture.is_some() && key.0 == CelestialBody::Earth;
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_show_day_night").as_ref(), if day_night_enabled { 1.0 } else { 0.0 });
            gl.uniform_3_f32(gl.get_uniform_location(self.program, "u_sun_dir").as_ref(), sun_dir[0], sun_dir[1], sun_dir[2]);
            gl.uniform_3_f32(gl.get_uniform_location(self.program, "u_bg_color").as_ref(), bg_color[0], bg_color[1], bg_color[2]);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_transparent_bg").as_ref(), 0.0);

            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        }
    }

    pub fn render_to_image(
        &self,
        gl: &glow::Context,
        key: (CelestialBody, Skin, TextureResolution),
        inv_rotation: &Matrix3<f64>,
        flattening: f64,
        size: usize,
    ) -> egui::ColorImage {
        let Some(texture) = self.textures.get(&key) else {
            return egui::ColorImage {
                size: [size, size],
                pixels: vec![egui::Color32::TRANSPARENT; size * size],
                source_size: egui::Vec2::ZERO,
            };
        };

        let ring_params = key.0.ring_params();
        let outer_ratio = ring_params.map(|(_, _, o)| o as f64).unwrap_or(1.0);
        let img_scale = if outer_ratio > 1.0 { outer_ratio } else { 1.0 };
        let fbo_size = (size as f64 * img_scale).ceil() as usize;
        let scale = (size as f32) / (fbo_size as f32);

        unsafe {
            let fbo = gl.create_framebuffer().expect("Cannot create FBO");
            let render_tex = gl.create_texture().expect("Cannot create render texture");

            gl.bind_texture(glow::TEXTURE_2D, Some(render_tex));
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA as i32,
                fbo_size as i32, fbo_size as i32, 0,
                glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D, Some(render_tex), 0,
            );

            gl.viewport(0, 0, fbo_size as i32, fbo_size as i32);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vertex_array));

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_texture").as_ref(), 0);

            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_clouds").as_ref(), 1);

            gl.active_texture(glow::TEXTURE2);
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_night").as_ref(), 2);

            gl.active_texture(glow::TEXTURE3);
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_detail").as_ref(), 3);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_use_detail").as_ref(), 0.0);

            gl.active_texture(glow::TEXTURE4);
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_stars").as_ref(), 4);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_show_stars").as_ref(), 0.0);

            gl.active_texture(glow::TEXTURE5);
            let has_rings = if let Some(rt) = self.ring_textures.get(&key.0) {
                gl.bind_texture(glow::TEXTURE_2D, Some(*rt));
                true
            } else {
                gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
                false
            };
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_ring_texture").as_ref(), 5);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_has_rings").as_ref(), if has_rings { 1.0 } else { 0.0 });
            let (ring_inner, ring_outer) = ring_params.map(|(_, i, o)| (i, o)).unwrap_or((0.0, 0.0));
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_ring_inner").as_ref(), ring_inner);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_ring_outer").as_ref(), ring_outer);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_adams_arc").as_ref(),
                if has_rings && key.0 == CelestialBody::Neptune { 1.0 } else { 0.0 });
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_epsilon_wobble").as_ref(),
                if has_rings && key.0 == CelestialBody::Uranus { 1.0 } else { 0.0 });

            let rot_data: [f32; 9] = [
                inv_rotation[(0, 0)] as f32, inv_rotation[(1, 0)] as f32, inv_rotation[(2, 0)] as f32,
                inv_rotation[(0, 1)] as f32, inv_rotation[(1, 1)] as f32, inv_rotation[(2, 1)] as f32,
                inv_rotation[(0, 2)] as f32, inv_rotation[(1, 2)] as f32, inv_rotation[(2, 2)] as f32,
            ];
            gl.uniform_matrix_3_f32_slice(
                gl.get_uniform_location(self.program, "u_inv_rotation").as_ref(),
                false, &rot_data,
            );

            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_flattening").as_ref(), flattening as f32);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_aspect").as_ref(), 1.0);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_scale").as_ref(), scale);
            gl.uniform_2_f32(gl.get_uniform_location(self.program, "u_uv_scale").as_ref(), 1.0, 1.0);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_atmosphere").as_ref(), 0.0);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_show_clouds").as_ref(), 0.0);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_show_day_night").as_ref(), 0.0);
            gl.uniform_3_f32(gl.get_uniform_location(self.program, "u_sun_dir").as_ref(), 0.0, 0.0, -1.0);
            gl.uniform_3_f32(gl.get_uniform_location(self.program, "u_bg_color").as_ref(), 0.0, 0.0, 0.0);
            gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_transparent_bg").as_ref(), 1.0);
            gl.uniform_4_f32(gl.get_uniform_location(self.program, "u_detail_bounds").as_ref(), 0.0, 0.0, 0.0, 0.0);

            gl.disable(glow::BLEND);

            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            gl.enable(glow::BLEND);

            let mut pixel_data = vec![0u8; fbo_size * fbo_size * 4];
            gl.read_pixels(
                0, 0, fbo_size as i32, fbo_size as i32,
                glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(pixel_data.as_mut_slice())),
            );

            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.delete_framebuffer(fbo);
            gl.delete_texture(render_tex);

            let mut pixels = Vec::with_capacity(fbo_size * fbo_size);
            for y in (0..fbo_size).rev() {
                for x in 0..fbo_size {
                    let idx = (y * fbo_size + x) * 4;
                    pixels.push(egui::Color32::from_rgba_unmultiplied(
                        pixel_data[idx],
                        pixel_data[idx + 1],
                        pixel_data[idx + 2],
                        pixel_data[idx + 3],
                    ));
                }
            }

            egui::ColorImage {
                size: [fbo_size, fbo_size],
                pixels,
                source_size: egui::Vec2::ZERO,
            }
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.program);
            gl.delete_vertex_array(self.vertex_array);
            for texture in self.textures.values() {
                gl.delete_texture(*texture);
            }
            for texture in self.cloud_textures.values() {
                gl.delete_texture(*texture);
            }
            if let Some(nt) = self.night_texture {
                gl.delete_texture(nt);
            }
            if let Some(st) = self.star_texture {
                gl.delete_texture(st);
            }
            if let Some(mw) = self.milky_way_texture {
                gl.delete_texture(mw);
            }
            for rt in self.ring_textures.values() {
                gl.delete_texture(*rt);
            }
        }
    }
}

pub struct MapRenderer {
    program: glow::Program,
    vertex_array: glow::VertexArray,
    earth_texture: Option<glow::Texture>,
    texture_key: Option<(CelestialBody, Skin, TextureResolution)>,
}

impl MapRenderer {
    pub fn new(gl: &glow::Context) -> Self {
        let shader_version = if cfg!(target_arch = "wasm32") {
            "#version 300 es"
        } else {
            "#version 330"
        };

        unsafe {
            let program = gl.create_program().expect("Cannot create program");

            let vertex_shader_source = r#"
                const vec2 verts[4] = vec2[4](
                    vec2(-1.0, -1.0),
                    vec2( 1.0, -1.0),
                    vec2(-1.0,  1.0),
                    vec2( 1.0,  1.0)
                );
                out vec2 v_uv;
                void main() {
                    v_uv = verts[gl_VertexID] * 0.5 + 0.5;
                    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
                }
            "#;

            let fragment_shader_source = r#"
                precision highp float;
                in vec2 v_uv;
                out vec4 out_color;

                uniform sampler2D u_earth;
                uniform int u_projection;
                uniform vec4 u_bounds;
                uniform vec4 u_peirce;
                uniform float u_peirce_inv_scale;

                const float PI = 3.14159265359;
                const float MOLL_S = 90.0;
                const float SIN_SX = 180.0;
                const float SIN_SY = 90.0;
                const float AE_S = 180.0;
                const float HAM_S = 90.0;
                const float CASS_S = 57.29577951;
                const float UTM_S = 57.29577951;
                const float LAEA_S = 90.0;

                vec3 ej(float u, float m) {
                    if (m < 1e-15) return vec3(sin(u), cos(u), 1.0);
                    if (m >= 1.0 - 1e-15) {
                        float t = tanh(u);
                        float s = 1.0 / cosh(u);
                        return vec3(t, s, s);
                    }
                    float aa[9], cc[9];
                    aa[0] = 1.0; cc[0] = sqrt(m);
                    float b = sqrt(1.0 - m);
                    int i = 0; float twon = 1.0;
                    for (int iter = 0; iter < 8; iter++) {
                        float ai = aa[i];
                        if (abs(cc[i] / ai) <= 1e-14) break;
                        i++;
                        cc[i] = (ai - b) / 2.0;
                        aa[i] = (ai + b) / 2.0;
                        b = sqrt(ai * b);
                        twon *= 2.0;
                    }
                    float phi = twon * aa[i] * u;
                    float bp = phi;
                    for (int iter = 0; iter < 9; iter++) {
                        bp = phi;
                        float t = cc[i] * sin(phi) / aa[i];
                        phi = (asin(clamp(t, -1.0, 1.0)) + phi) / 2.0;
                        i--;
                        if (i == 0) break;
                    }
                    float cn = cos(phi);
                    float dc = cos(phi - bp);
                    float dn = abs(dc) > 1e-30 ? cn / dc : 1.0;
                    return vec3(sin(phi), cn, dn);
                }

                void eji(float u, float v, float m,
                         out vec2 sn, out vec2 cn, out vec2 dn) {
                    if (abs(u) < 1e-15) {
                        vec3 j = ej(v, 1.0 - m);
                        sn = vec2(0.0, j.x / j.y);
                        cn = vec2(1.0 / j.y, 0.0);
                        dn = vec2(j.z / j.y, 0.0);
                        return;
                    }
                    vec3 ja = ej(u, m);
                    if (abs(v) < 1e-15) {
                        sn = vec2(ja.x, 0.0);
                        cn = vec2(ja.y, 0.0);
                        dn = vec2(ja.z, 0.0);
                        return;
                    }
                    vec3 jb = ej(v, 1.0 - m);
                    float d = jb.y * jb.y + m * ja.x * ja.x * jb.x * jb.x;
                    sn = vec2(ja.x * jb.z / d,
                              ja.y * ja.z * jb.x * jb.y / d);
                    cn = vec2(ja.y * jb.y / d,
                             -ja.x * ja.z * jb.x * jb.z / d);
                    dn = vec2(ja.z * jb.y * jb.z / d,
                             -m * ja.x * ja.y * jb.x / d);
                }

                vec2 guyou_inv(float x, float y,
                               float k_, float m, float bk) {
                    vec2 sn, cn, dn;
                    eji(0.5 * bk - y, -x, m, sn, cn, dn);
                    float d = cn.x * cn.x + cn.y * cn.y;
                    if (d < 1e-30) return vec2(0.0, 0.0);
                    float tr = (sn.x * cn.x + sn.y * cn.y) / d;
                    float ti = (sn.y * cn.x - sn.x * cn.y) / d;
                    float lam = -atan(ti, tr);
                    float la = k_ * (tr * tr + ti * ti);
                    float phi = 2.0 * atan(exp(-0.5 * log(la)))
                                - PI / 2.0;
                    return vec2(lam, phi);
                }

                vec2 inv_proj(int p, float x, float y) {
                    if (p == 0) {
                        if (abs(x) > 180.0 || abs(y) > 90.0)
                            return vec2(-999.0);
                        return vec2(y, x);
                    }
                    if (p == 1) {
                        float lr = 2.0*atan(exp(radians(y)))-PI/2.0;
                        float ld = degrees(lr);
                        if (abs(ld)>85.0||abs(x)>180.0)
                            return vec2(-999.0);
                        return vec2(ld, x);
                    }
                    if (p == 2) {
                        float xn=x/MOLL_S, yn=y/MOLL_S;
                        float st=yn/sqrt(2.0);
                        if(abs(st)>1.0) return vec2(-999.0);
                        float th=asin(st), ct=cos(th);
                        if(abs(ct)<1e-10) return vec2(-999.0);
                        float lon=PI*xn/(2.0*sqrt(2.0)*ct);
                        if(abs(lon)>PI) return vec2(-999.0);
                        float sl=(2.0*th+sin(2.0*th))/PI;
                        if(abs(sl)>1.0) return vec2(-999.0);
                        return vec2(degrees(asin(sl)),degrees(lon));
                    }
                    if (p == 3) {
                        float lr=y/SIN_SY*(PI/2.0);
                        if(abs(lr)>PI/2.0) return vec2(-999.0);
                        float cl=cos(lr);
                        if(abs(cl)<1e-10) return vec2(-999.0);
                        float lon=x/SIN_SX*PI/cl;
                        if(abs(lon)>PI) return vec2(-999.0);
                        return vec2(degrees(lr),degrees(lon));
                    }
                    if (p == 4) {
                        float xn=x/AE_S*PI, yn=y/AE_S*PI;
                        float c=sqrt(xn*xn+yn*yn);
                        if(c>PI) return vec2(-999.0);
                        return vec2(degrees(PI/2.0-c),
                                    degrees(atan(xn,-yn)));
                    }
                    if (p == 5) {
                        float xn=x/HAM_S, yn=y/HAM_S;
                        float z2=1.0-(xn/4.0)*(xn/4.0)
                                     -(yn/2.0)*(yn/2.0);
                        if(z2<0.0) return vec2(-999.0);
                        float z=sqrt(z2);
                        float lon=2.0*atan(z*xn,
                                           2.0*(2.0*z2-1.0));
                        float sl=z*yn;
                        if(abs(sl)>1.0) return vec2(-999.0);
                        if(abs(lon)>PI) return vec2(-999.0);
                        return vec2(degrees(asin(sl)),
                                    degrees(lon));
                    }
                    if (p == 6) {
                        float xr=radians(x), yr=radians(y);
                        if(abs(yr)>PI/2.0) return vec2(-999.0);
                        if(abs(yr)<=PI/4.0) {
                            float sp=8.0*yr/(3.0*PI);
                            if(abs(sp)>1.0) return vec2(-999.0);
                            if(abs(xr)>PI) return vec2(-999.0);
                            return vec2(degrees(asin(sp)),
                                        degrees(xr));
                        }
                        float sv=sign(yr);
                        float sig=2.0-4.0*abs(yr)/PI;
                        if(sig<1e-10) return vec2(sv*90.0,0.0);
                        float step=PI/2.0;
                        float lc=round((xr-PI/4.0)/step)*step
                                  +PI/4.0;
                        float lam=lc+(xr-lc)/sig;
                        if(abs(lam-lc)>step/2.0+0.001)
                            return vec2(-999.0);
                        if(abs(lam)>PI+0.001)
                            return vec2(-999.0);
                        float sp=1.0-sig*sig/3.0;
                        if(sp>1.0) return vec2(-999.0);
                        return vec2(sv*degrees(asin(sp)),
                                    clamp(degrees(lam),
                                          -180.0,180.0));
                    }
                    if (p == 7) {
                        float xr=x/CASS_S, yr=y/CASS_S;
                        if(abs(xr)>PI/2.0)return vec2(-999.0);
                        return vec2(degrees(asin(sin(yr)*cos(xr))),
                                    degrees(atan(tan(xr),cos(yr))));
                    }
                    if (p == 8) {
                        float zone=clamp(floor((x+180.0)/6.0),
                                         0.0,59.0);
                        float cm=zone*6.0-177.0;
                        float xr=(x-cm)/UTM_S, yr=y/UTM_S;
                        float sl=sin(yr)/cosh(xr);
                        if(abs(sl)>1.0) return vec2(-999.0);
                        float dl=atan(sinh(xr),cos(yr));
                        if(abs(dl)>radians(3.0)+0.001)
                            return vec2(-999.0);
                        return vec2(degrees(asin(sl)),
                                    cm+degrees(dl));
                    }
                    if (p == 9) {
                        float xn=x/LAEA_S, yn=y/LAEA_S;
                        float rho=sqrt(xn*xn+yn*yn);
                        if(rho>2.0) return vec2(-999.0);
                        if(rho<1e-10) return vec2(0.0,0.0);
                        float c=2.0*asin(rho/2.0);
                        return vec2(degrees(asin(yn*sin(c)/rho)),
                                    degrees(atan(xn*sin(c),
                                                 rho*cos(c))));
                    }
                    if (p == 10) {
                        float s12 = sqrt(0.5);
                        float pm = u_peirce.x;
                        float pk = u_peirce.y;
                        float pbk = u_peirce.z;
                        float pdx = u_peirce.w;
                        float xx = x * u_peirce_inv_scale;
                        float yy = y * u_peirce_inv_scale;
                        float gx = (xx + yy) * s12;
                        float gy = (yy - xx) * s12;
                        float hd = 0.5 * pdx;
                        bool front = abs(gx)<hd+0.001
                                  && abs(gy)<hd+0.001;
                        if (front) {
                            vec2 ll = guyou_inv(gx, gy,
                                               pk, pm, pbk);
                            float la = degrees(ll.y);
                            float lo = degrees(ll.x);
                            if (abs(la)<=90.0&&abs(lo)<=180.0)
                                return vec2(la, lo);
                            return vec2(-999.0);
                        }
                        float dd = pdx * s12;
                        float ss = ((xx>0.0)!=(yy>0.0))
                                   ? -1.0 : 1.0;
                        float x1 = -ss*xx
                            + (yy>0.0?1.0:-1.0)*dd;
                        float y1 = -ss*yy
                            + (xx>0.0?1.0:-1.0)*dd;
                        float gx2 = (-x1-y1)*s12;
                        float gy2 = (x1-y1)*s12;
                        vec2 ll = guyou_inv(gx2, gy2,
                                           pk, pm, pbk);
                        float lo = degrees(ll.x)
                            + (gx2>0.0 ? 180.0 : -180.0);
                        float la = degrees(ll.y);
                        if (abs(la)<=90.0&&abs(lo)<=180.0)
                            return vec2(la, lo);
                        return vec2(-999.0);
                    }
                    return vec2(-999.0);
                }

                void main() {
                    float x = mix(u_bounds.x, u_bounds.y, v_uv.x);
                    float y = mix(u_bounds.z, u_bounds.w, v_uv.y);
                    vec2 ll = inv_proj(u_projection, x, y);
                    if (ll.x < -900.0) {
                        out_color = vec4(0.0);
                        return;
                    }
                    float u = (ll.y + 180.0) / 360.0;
                    float v = (90.0 - ll.x) / 180.0;
                    out_color = texture(u_earth, vec2(u, v));
                }
            "#;

            let shader_sources = [
                (glow::VERTEX_SHADER, vertex_shader_source),
                (glow::FRAGMENT_SHADER, fragment_shader_source),
            ];

            let shaders: Vec<_> = shader_sources
                .iter()
                .map(|(shader_type, shader_source)| {
                    let shader = gl
                        .create_shader(*shader_type)
                        .expect("Cannot create shader");
                    gl.shader_source(
                        shader,
                        &format!("{shader_version}\n{shader_source}"),
                    );
                    gl.compile_shader(shader);
                    assert!(
                        gl.get_shader_compile_status(shader),
                        "Failed to compile map shader: {}",
                        gl.get_shader_info_log(shader)
                    );
                    gl.attach_shader(program, shader);
                    shader
                })
                .collect();

            gl.link_program(program);
            assert!(
                gl.get_program_link_status(program),
                "Failed to link map program: {}",
                gl.get_program_info_log(program)
            );

            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }

            let vertex_array = gl
                .create_vertex_array()
                .expect("Cannot create vertex array");

            Self {
                program,
                vertex_array,
                earth_texture: None,
                texture_key: None,
            }
        }
    }

    pub fn upload_earth_texture(
        &mut self,
        gl: &glow::Context,
        key: (CelestialBody, Skin, TextureResolution),
        earth_tex: &crate::texture::EarthTexture,
    ) {
        if self.texture_key == Some(key) {
            return;
        }
        if let Some(old) = self.earth_texture.take() {
            unsafe { gl.delete_texture(old); }
        }
        unsafe {
            let texture = gl
                .create_texture()
                .expect("Cannot create texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            let pixels: Vec<u8> = earth_tex
                .pixels
                .iter()
                .flat_map(|&[r, g, b]| [r, g, b])
                .collect();
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB as i32,
                earth_tex.width as i32,
                earth_tex.height as i32,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pixels)),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::REPEAT as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            self.earth_texture = Some(texture);
            self.texture_key = Some(key);
        }
    }

    #[allow(dead_code)]
    pub fn has_texture(&self) -> bool {
        self.earth_texture.is_some()
    }

    #[allow(dead_code)]
    pub fn invalidate_texture(&mut self, gl: &glow::Context) {
        if let Some(tex) = self.earth_texture.take() {
            unsafe { gl.delete_texture(tex); }
        }
    }

    pub fn paint(
        &self,
        gl: &glow::Context,
        proj_id: i32,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
    ) {
        let Some(earth_tex) = self.earth_texture else {
            return;
        };

        let pc = crate::projection::peirce_const();

        unsafe {
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vertex_array));

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(earth_tex));
            gl.uniform_1_i32(
                gl.get_uniform_location(self.program, "u_earth")
                    .as_ref(),
                0,
            );

            gl.uniform_1_i32(
                gl.get_uniform_location(
                    self.program, "u_projection",
                ).as_ref(),
                proj_id,
            );
            gl.uniform_4_f32(
                gl.get_uniform_location(
                    self.program, "u_bounds",
                ).as_ref(),
                xmin as f32, xmax as f32, ymin as f32, ymax as f32,
            );
            gl.uniform_4_f32(
                gl.get_uniform_location(
                    self.program, "u_peirce",
                ).as_ref(),
                pc.m as f32, pc.k_ as f32,
                pc.big_k as f32, pc.dx as f32,
            );
            gl.uniform_1_f32(
                gl.get_uniform_location(
                    self.program, "u_peirce_inv_scale",
                ).as_ref(),
                pc.inv_scale as f32,
            );

            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.program);
            gl.delete_vertex_array(self.vertex_array);
            if let Some(tex) = self.earth_texture {
                gl.delete_texture(tex);
            }
        }
    }
}
