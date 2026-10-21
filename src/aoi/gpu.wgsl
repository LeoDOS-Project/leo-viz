struct Uniforms {
    rotation: array<vec4<f32>, 3>,
    screen: vec4<f32>,
    viewport: vec4<f32>,
    params: vec4<f32>,
}
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) local: vec2<f32>,
    @location(2) @interpolate(flat) half_width: f32,
}

const CORNERS = array<vec2<f32>, 6>(
    vec2(0.0, -1.0), vec2(0.0, 1.0), vec2(1.0, -1.0),
    vec2(1.0, -1.0), vec2(0.0, 1.0), vec2(1.0, 1.0),
);

fn project(p: vec3<f32>) -> vec3<f32> {
    let surface = vec3(p.x, p.y * u.params.x, p.z);
    let normal = vec3(p.x, p.y / u.params.x, p.z);
    return vec3(
        u.screen.xy + vec2(dot(u.rotation[0].xyz, surface), dot(u.rotation[1].xyz, surface)) * u.screen.zw,
        dot(u.rotation[2].xyz, normal),
    );
}

fn clipped() -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4(2.0, 2.0, 0.0, 1.0);
    output.color = vec4(0.0);
    output.local = vec2(0.0);
    output.half_width = 0.0;
    return output;
}

fn output_vertex(position: vec2<f32>, local: vec2<f32>, color: vec4<f32>, half_width: f32) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4(position / u.viewport.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
    output.color = color;
    output.local = local;
    output.half_width = half_width;
    return output;
}

fn visible(anchor: vec3<f32>, radius: f32) -> bool {
    let center = project(anchor);
    let extent = radius * u.params.y + 3.0;
    return center.z >= -radius / u.params.x &&
        all(center.xy >= vec2(-extent)) && all(center.xy <= u.viewport.xy + vec2(extent));
}

@vertex
fn line_vertex(
    @builtin(vertex_index) vertex: u32,
    @location(0) start: vec3<f32>,
    @location(1) end: vec3<f32>,
    @location(2) bound: vec4<f32>,
    @location(3) color: vec4<f32>,
    @location(4) scales: vec2<f32>,
) -> VertexOutput {
    if bound.w * u.params.y < 3.0 || u.params.y <= scales.x || u.params.y > scales.y || !visible(bound.xyz, bound.w) {
        return clipped();
    }
    let pa = project(start);
    let pb = project(end);
    if pa.z < 0.0 && pb.z < 0.0 { return clipped(); }
    var a = pa.xy;
    var b = pb.xy;
    if (pa.z < 0.0) != (pb.z < 0.0) {
        let crossing = mix(a, b, pa.z / (pa.z - pb.z));
        if pa.z < 0.0 { a = crossing; } else { b = crossing; }
    }
    let delta = b - a;
    let length = length(delta);
    if length < 0.0001 { return clipped(); }
    let corner = CORNERS[vertex];
    let half_width = 0.6;
    let outer = half_width + 0.5 / u.viewport.z;
    let perpendicular = vec2(-delta.y, delta.x) / length;
    let position = mix(a, b, corner.x) + perpendicular * corner.y * outer;
    return output_vertex(position, vec2(0.0, corner.y * outer), color, half_width);
}

@vertex
fn dot_vertex(
    @builtin(vertex_index) vertex: u32,
    @location(0) anchor: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
) -> VertexOutput {
    let center = project(anchor);
    if radius * u.params.y >= 3.0 || center.z < 0.0 || !visible(anchor, radius) { return clipped(); }
    let corner = CORNERS[vertex] * vec2(2.0, 1.0) - vec2(1.0, 0.0);
    let half_width = 1.25;
    let local = corner * (half_width + 0.5 / u.viewport.z);
    return output_vertex(center.xy + local, local, color, half_width);
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    let distance = max(abs(input.local.x), abs(input.local.y));
    let aa = 0.5 / u.viewport.z;
    let coverage = 1.0 - smoothstep(input.half_width - aa, input.half_width + aa, distance);
    let alpha = input.color.a * u.viewport.w * coverage;
    var rgb = input.color.rgb;
    if u.params.z > 0.5 {
        rgb = select(rgb / 12.92, pow((rgb + vec3(0.055)) / 1.055, vec3(2.4)), rgb > vec3(0.04045));
    }
    return vec4(rgb * alpha, alpha);
}
