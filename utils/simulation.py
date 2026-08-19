import os
import math
import torch
import numpy as np
import imageio.v2 as imageio
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from utils.math import get_rotation_matrix, matrix_to_quaternion, q_mul
from utils.camera import get_world_to_view_matrix, getProjectionMatrix
from utils.physics import compute_cov3d_deformed
import utils.physics_engine as phys

def run_simulation(
    model_data,
    E_param,
    grav_param,
    rot_x,
    rot_y,
    rot_z,
    model_scale,
    model_y_offset,
    cam_pos,
    target_pos,
    material,
    hardening,
    friction_deg,
    yield_stress,
    floor_friction,
    crusher_mass_val,
    enable_crusher,
    fracture_threshold,
):
    print(f"\n[SIMULATION] Material: {phys.MATERIAL_NAMES[material]}")
    print("[SIMULATION] Applying transformations and starting video render...")

    pos_local = model_data["pos_local_phys"]
    base_offset = model_data["base_offset"]
    render_cols_np = model_data["render_cols_np"]
    render_shs_np = model_data["render_shs_np"]
    render_scale_np = model_data["render_scale_np"]
    render_rot_np = model_data["render_rot_np"]
    render_opac_np = model_data["render_opac_np"]
    skin_idx_np = model_data["skin_idx_np"]
    local_offset_canonical_np = model_data["local_offset_canonical_np"]

    R = get_rotation_matrix(rot_x, rot_y, rot_z)
    initial_pos = (
        (R @ (pos_local * model_scale).T).T
        + base_offset
        + np.array([0.0, model_y_offset, 0.0], dtype=np.float32)
    )

    if enable_crusher:
        phys.enable_crusher_ti[None] = 1
        phys.crusher_y_ti[None] = max(np.min(initial_pos[:, 1]) - 0.05, 0.05)
    else:
        phys.enable_crusher_ti[None] = 0
        
    # Clamp all initial positions strictly inside the grid to prevent Taichi out-of-bounds segfaults
    initial_pos = np.clip(initial_pos, 0.05, 0.85)
        
    phys.active_N[None] = model_data["N"]
    phys.active_N_total[None] = model_data["N"]
    phys.init_data(initial_pos)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mu = E_param / (2 * (1 + phys.nu))
    lam = E_param * phys.nu / ((1 + phys.nu) * (1 - 2 * phys.nu))

    # --- Data for high-resolution rendering (M gaussians) ---
    t_colors = torch.tensor(render_cols_np, dtype=torch.float32, device=device)
    t_shs = torch.tensor(render_shs_np, dtype=torch.float32, device=device)
    t_scale = torch.tensor(
        render_scale_np * model_scale, dtype=torch.float32, device=device
    )

    model_q = matrix_to_quaternion(R)
    rotated_rot = q_mul(model_q, render_rot_np)
    t_rot = torch.tensor(rotated_rot, dtype=torch.float32, device=device)

    t_opac = torch.tensor(render_opac_np, dtype=torch.float32, device=device).unsqueeze(1)

    # --- Skinning data: guide particle index + local offset in world frame ---
    t_skin_idx = torch.tensor(skin_idx_np, dtype=torch.long, device=device)

    # The local offset must be rotated/scaled like the positions (but NOT translated):
    # it is a difference vector, so it transforms only with R and model_scale.
    local_offset_world_np = (R @ (local_offset_canonical_np * model_scale).T).T
    t_local_offset = torch.tensor(
        local_offset_world_np, dtype=torch.float32, device=device
    )

    W, H = 1280, 720
    fovY = 60.0 * math.pi / 180.0
    tan_fovy = math.tan(fovY / 2)
    tan_fovx = tan_fovy * (W / H)
    fovX = 2.0 * math.atan(tan_fovx)

    viewmatrix = get_world_to_view_matrix(cam_pos, target_pos, device)
    projmatrix = getProjectionMatrix(0.01, 100.0, fovX, fovY, device)
    full_proj = viewmatrix @ projmatrix

    bg_color = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device)

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tan_fovx,
        tanfovy=tan_fovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_pos,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings)
    os.makedirs("frames_video", exist_ok=True)

    MAX_FRAMES = 300
    for frame_idx in range(MAX_FRAMES):
        if enable_crusher:
            if phys.crusher_y_ti[None] < 0.6:  # Stop midway (floor is at 0.9)
                phys.crusher_y_ti[None] += 0.5 * (phys.dt * 100)  # 0.5 m/s downwards
        for _ in range(100):
            phys.mpm_substep(
                mu,
                lam,
                grav_param,
                material,
                hardening,
                friction_deg,
                yield_stress,
                floor_friction,
                crusher_mass_val,
                fracture_threshold,
            )

        # "Lightweight" physical state (N particles)
        t_pos_phys = torch.from_numpy(phys.x.to_numpy()).to(device)
        t_F_phys = torch.from_numpy(phys.F.to_numpy()).to(device)

        # --- Skinning: rebuild the dense M gaussians starting from the N particles ---
        x_g = t_pos_phys[t_skin_idx]  # (M, 3)
        F_g = t_F_phys[t_skin_idx]  # (M, 3, 3)
        t_pos_render = x_g + torch.bmm(F_g, t_local_offset.unsqueeze(-1)).squeeze(-1)

        t_cov3d = compute_cov3d_deformed(t_scale, t_rot, F_g)

        image, _ = rasterizer(
            means3D=t_pos_render,
            means2D=torch.zeros_like(t_pos_render, device=device),
            shs=t_shs,
            colors_precomp=None,
            opacities=t_opac,
            scales=None,
            rotations=None,
            cov3D_precomp=t_cov3d,
        )

        img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        img_np = np.flipud(img_np)  # Flip vertically because the world Y-axis points downwards
        imageio.imwrite(f"frames_video/frame_{frame_idx:04d}.png", img_np)

        if frame_idx % 30 == 0:
            print(f"Rendered frame {frame_idx}/{MAX_FRAMES}...")

    print("Generating MP4 video...")
    writer = imageio.get_writer("simulazione_fotorealistica.mp4", fps=60)
    for i in range(MAX_FRAMES):
        filename = f"frames_video/frame_{i:04d}.png"
        if os.path.exists(filename):
            writer.append_data(imageio.imread(filename))
    writer.close()
    print("Video 'simulazione_fotorealistica.mp4' saved successfully!")
