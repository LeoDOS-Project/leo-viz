struct Camera {
    view_projection: mat4x4f,
    eye: vec4f,
    options: vec4f,
};
struct Material {
    color: vec4f,
    surface: vec4f,
    normal: vec4f,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var base_texture: texture_2d<f32>;
@group(1) @binding(1) var roughness_texture: texture_2d<f32>;
@group(1) @binding(2) var normal_texture: texture_2d<f32>;
@group(1) @binding(3) var texture_sampler: sampler;
@group(1) @binding(4) var<uniform> material: Material;

struct VertexOutput {
    @builtin(position) clip: vec4f,
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) color: vec4f,
    @location(4) @interpolate(flat) part: u32,
    @location(5) normal_uv: vec2f,
    @location(6) roughness_uv: vec2f,
};
@vertex fn vs(
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) normal_uv: vec2f,
    @location(4) roughness_uv: vec2f,
    @location(5) color: vec4f,
    @location(6) offset: vec3f,
    @location(7) part: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = position + offset * camera.options.x;
    out.clip = camera.view_projection * vec4f(out.position, 1.0);
    out.normal = normal;
    out.uv = uv;
    out.color = color;
    out.part = part;
    out.normal_uv = normal_uv;
    out.roughness_uv = roughness_uv;
    return out;
}

fn fresnel(cosine: f32, f0: vec3f) -> vec3f {
    return f0 + (vec3f(1.0)-f0)*pow(1.0-cosine, 5.0);
}
fn light(n: vec3f, v: vec3f, l: vec3f, color: vec3f, metallic: f32, roughness: f32) -> vec3f {
    let h = normalize(v+l);
    let nv = max(dot(n,v), 0.001);
    let nl = max(dot(n,l), 0.001);
    let nh = max(dot(n,h), 0.0);
    let a = roughness*roughness;
    let a2 = a*a;
    let d = a2/(3.14159265*pow(nh*nh*(a2-1.0)+1.0, 2.0));
    let k = pow(roughness+1.0,2.0)/8.0;
    let g = nv/(nv*(1.0-k)+k) * nl/(nl*(1.0-k)+k);
    let f = fresnel(max(dot(h,v), 0.0), mix(vec3f(0.04), color, metallic));
    let specular = d*g*f/max(4.0*nv*nl,0.001);
    let diffuse = (vec3f(1.0)-f)*(1.0-metallic)*color/3.14159265;
    return (diffuse+specular)*nl;
}
@fragment fn fs(in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let selected = camera.options.y >= 0.0 && u32(camera.options.y) == in.part;
    let hovered = camera.options.w >= 0.0 && u32(camera.options.w) == in.part;
    if camera.options.z > 0.5 && !selected { discard; }
    var base = textureSample(base_texture, texture_sampler, in.uv)*material.color*in.color;
    if material.normal.z > 0.5 && material.normal.z < 1.5 {
        base = vec4f(0.58,0.48,0.10,1.0);
    }
    if material.normal.z > 1.5 {
        base = vec4f(vec3f(0.035),min(base.a*4.0,1.0));
    }
    if base.a < material.surface.z { discard; }
    let mr = textureSample(roughness_texture, texture_sampler, in.roughness_uv);
    var metallic = clamp(material.surface.x*mr.b,0.0,1.0);
    var roughness = clamp(material.surface.y*mr.g,0.08,1.0);
    if material.normal.z > 0.5 {
        metallic = 0.15;
        roughness = 0.7;
    }
    var n = normalize(in.normal);
    if !front { n = -n; }
    let mapped = textureSample(normal_texture,texture_sampler,in.normal_uv).xyz*2.0-vec3f(1.0);
    let dp1 = dpdx(in.position);
    let dp2 = dpdy(in.position);
    let uv1 = dpdx(in.normal_uv);
    let uv2 = dpdy(in.normal_uv);
    let p2 = cross(dp2,n);
    let p1 = cross(n,dp1);
    let tangent = p2*uv1.x+p1*uv2.x;
    let bitangent = p2*uv1.y+p1*uv2.y;
    let inverse_scale = inverseSqrt(max(max(dot(tangent,tangent),dot(bitangent,bitangent)),0.0000001));
    n = normalize(mat3x3f(tangent*inverse_scale,bitangent*inverse_scale,n)*vec3f(mapped.xy*material.normal.x,mapped.z));
    let v = normalize(camera.eye.xyz-in.position);
    // Studio illumination is independent of the orbital simulation.
    let key = normalize(camera.eye.xyz + vec3f(-1.8,3.5,1.0));
    let fill = normalize(vec3f(-3.0,1.0,-2.0));
    var color = light(n,v,key,base.rgb,metallic,roughness)*3.0;
    color += light(n,v,fill,base.rgb,metallic,roughness)*1.0;
    let sky = mix(vec3f(0.20,0.22,0.25),vec3f(0.66,0.70,0.74),n.y*0.5+0.5);
    color += base.rgb*sky*(0.48*(1.0-metallic)+0.33*metallic);
    color += fresnel(max(dot(n,v),0.0),mix(vec3f(0.04),base.rgb,metallic))*sky*0.30;
    color = clamp((color*(2.51*color+vec3f(0.03)))/(color*(2.43*color+vec3f(0.59))+vec3f(0.14)),vec3f(0.0),vec3f(1.0));
    if hovered {
        let rim = pow(1.0-abs(dot(n,v)),2.0);
        color = mix(color,vec3f(0.10,0.65,1.0),0.30+0.35*rim);
    }
    if !front && material.normal.y < 0.5 { discard; }
    // Keep the source perimeter truss, suppressing the dense illustrative web inside it.
    if material.normal.z > 1.5 && length((in.uv-vec2f(0.5))*2.0) < material.normal.w { discard; }
    return vec4f(color,select(1.0,base.a,material.surface.w > 0.5));
}
