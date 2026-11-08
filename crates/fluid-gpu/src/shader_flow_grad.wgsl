// Central-difference gradient planes of one luma level ('fluid-flow' gradient_planes).
@group(0) @binding(0) var<storage, read> luma: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.dims.x;
    let h = params.dims.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let x = gid.x;
    let y = gid.y;
    let xl = x - min(x, 1u);
    let xr = min(x + 1u, w - 1u);
    let yu = y - min(y, 1u);
    let yd = min(y + 1u, h - 1u);
    let i = y * w + x;
    grad_x[i] = 0.5 * (luma[y * w + xr] - luma[y * w + xl]);
    grad_y[i] = 0.5 * (luma[yd * w + x] - luma[yu * w + x]);
}
