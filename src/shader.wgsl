struct Camera {
    view_proj : mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> u_camera : Camera;

struct Instance {
    model : mat4x4<f32>,
    color : vec4<f32>,
};

@group(1) @binding(0)
var<uniform> u_instance : Instance;

struct VsIn {
    @location(0) position : vec3<f32>,
};

struct VsOut {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>,
};

@vertex
fn vs_main(input : VsIn) -> VsOut {
    var out : VsOut;
    let world_pos = u_instance.model * vec4<f32>(input.position, 1.0);
    out.position = u_camera.view_proj * world_pos;
    out.color = u_instance.color;
    return out;
}

@fragment
fn fs_main(input : VsOut) -> @location(0) vec4<f32> {
    return input.color;
}

