import torch
import numpy as np
import taichi as ti
from plyfile import PlyData
import math
import os
import datetime
import imageio.v2 as imageio
import open3d as o3d
from scipy.spatial import cKDTree
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

ti.init(arch=ti.cuda, device_memory_fraction=0.2)

from utils.ply import load_3dgs_ply
from utils.camera import get_world_to_view_matrix, getProjectionMatrix, FreeCamera
from utils.video_exporter import VideoExporter
import sys

# ==========================================
# 4. MAIN & EFFECTS EDITOR
# ==========================================

def build_raycasting_scene(pos_np):
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos_np)
    
    # Downsample and estimate normals
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # Create proxy mesh using Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    
    # Convert to tensor mesh and add to raycasting scene
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)
    return scene

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_3dgs_ply("modello.ply")
    if data is None:
        return
    pos_np, shs_np, opac_np, scale_np, rot_np = data["pos"], data["shs"], data["opac"], data["scale"], data["rot"]

    center_np = pos_np.mean(axis=0)
    extent = float((pos_np.max(axis=0) - pos_np.min(axis=0)).max())

    # Initialize Collision Structures
    ray_scene = build_raycasting_scene(pos_np)
    print("[INFO] Building KD-Tree for fast gaussian search (laser modifications)...")
    kdtree = cKDTree(pos_np)
    print("[OK] KD-Tree ready.")

    # Transfer data to GPU
    t_means3D = torch.tensor(pos_np, device=device)
    t_shs_orig = torch.tensor(shs_np, device=device)  # BACKUP for RESET
    t_shs = t_shs_orig.clone()  # DYNAMIC TENSOR
    t_opac = torch.tensor(opac_np, device=device).unsqueeze(1)
    t_scales = torch.tensor(scale_np, device=device)
    t_rots = torch.tensor(rot_np, device=device)

    W, H = 1280, 720
    window = ti.ui.Window(
        "Gaussian Effects Editor (Laser & Screenshot)", res=(W, H), show_window=True
    )
    canvas = window.get_canvas()
    gui = window.get_gui()
    cam_pos_np = np.array([0.5, 0.4, 1.3], dtype=np.float32)
    target_np = np.array([0.5, 0.4, 0.5], dtype=np.float32)
    camera = FreeCamera(position=cam_pos_np, target=target_np)

    video_exporter = VideoExporter(output_dir="results")

    # UI State
    enable_laser = False
    laser_radius = extent * 0.03
    laser_power = 0.5
    SH_C0 = 0.28209479177387814

    print("\n[INFO] Editor Startup Complete!")
    print(" - LMB (Left Mouse Button) + Drag: Rotate Camera.")
    print(" - WASD: Move camera forward/left/back/right.")
    print(" - UP / DOWN: Move camera up and down.")

    while window.running:
        speed = extent * 0.01

        cam_pos_np, target_np = camera.update(window, speed)
        cam_pos = torch.tensor(cam_pos_np, dtype=torch.float32, device=device)
        t_target = torch.tensor(target_np, dtype=torch.float32, device=device)

        viewmatrix, projmatrix, _, _, tan_fovx, tan_fovy = camera.get_matrices(
            device, W, H, znear=0.01, zfar=extent * 10.0
        )
        fovY = camera.fovY
        fovX = 2.0 * math.atan(tan_fovx)
        
        t_hit = math.inf
        # ==========================================
        # LASER EFFECT MANAGEMENT
        # ==========================================
        if enable_laser:
            # Map mouse to screen coordinates [-1, 1]
            curr_mouse = window.get_cursor_pos()
            mx, my = curr_mouse[0], curr_mouse[1]
            nx = 2.0 * mx - 1.0
            ny = 2.0 * my - 1.0

            fwd_full = target_np - cam_pos_np
            fwd_full /= np.linalg.norm(fwd_full) + 1e-6
            right_full = np.cross(fwd_full, np.array([0, -1, 0]))
            right_full /= np.linalg.norm(right_full) + 1e-6
            up_full = np.cross(right_full, fwd_full)

            # Reconstruct the 3D ray direction based on camera FoV
            ray_dir = right_full * (nx * tan_fovx) + up_full * (ny * tan_fovy) + fwd_full
            ray_dir /= np.linalg.norm(ray_dir)

            ray_tensor = o3d.core.Tensor(
                [
                    [
                        cam_pos_np[0],
                        cam_pos_np[1],
                        cam_pos_np[2],
                        ray_dir[0],
                        ray_dir[1],
                        ray_dir[2],
                    ]
                ],
                dtype=o3d.core.float32,
            )

            # Cast ray against proxy scene
            ans = ray_scene.cast_rays(ray_tensor)
            t_hit = ans["t_hit"][0].item()

            if not math.isinf(t_hit):
                hit_pos = cam_pos_np + ray_dir * t_hit

                # 2. BURN THE MODEL IF LEFT OR MIDDLE CLICK IS HELD
                if window.is_pressed(ti.ui.LMB) or window.is_pressed(ti.ui.MMB):
                    indices = kdtree.query_ball_point(hit_pos, r=laser_radius)
                    if len(indices) > 0:
                        with torch.no_grad():
                            # Distance from impact center for falloff blending
                            dist_tensor = torch.norm(
                                t_means3D[indices]
                                - torch.tensor(
                                    hit_pos, dtype=torch.float32, device=device
                                ),
                                dim=1,
                            )
                            intensity = torch.clamp(
                                1.0 - (dist_tensor / laser_radius), 0.0, 1.0
                            )

                            # Add noise for a realistic ash effect
                            noise = torch.rand_like(intensity) * 0.4

                            # Burn factor for a single frame (accumulative)
                            burn_factor = (
                                torch.clamp(
                                    (intensity + noise) * laser_power * 0.15, 0.0, 1.0
                                )
                                .unsqueeze(1)
                                .to(torch.float32)
                            )

                            # Dark charcoal color
                            charcoal_rgb = torch.tensor(
                                [0.05, 0.03, 0.02], device=device, dtype=torch.float32
                            )
                            charcoal_dc = (charcoal_rgb - 0.5) / SH_C0

                            # Blend the current color towards charcoal
                            current_dc = t_shs[indices, 0, :]
                            t_shs[indices, 0, :] = torch.lerp(
                                current_dc, charcoal_dc, burn_factor
                            )

                            # Gradually zero out reflections (opacitying the burn)
                            if t_shs.shape[1] > 1:
                                t_shs[indices, 1:, :] = torch.lerp(
                                    t_shs[indices, 1:, :],
                                    torch.zeros_like(t_shs[indices, 1:, :]),
                                    burn_factor.unsqueeze(2),
                                )

        # ==========================================
        # RENDERING
        # ==========================================
        cam_pos = torch.tensor(cam_pos_np, dtype=torch.float32, device=device)
        t_target_tch = torch.tensor(target_np, dtype=torch.float32, device=device)

        full_proj = viewmatrix @ projmatrix

        raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tan_fovx,
            tanfovy=tan_fovy,
            bg=torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device),
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=full_proj,
            sh_degree=3,
            campos=cam_pos,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings)

        with torch.no_grad():
            image, _ = rasterizer(
                means3D=t_means3D,
                means2D=torch.zeros_like(t_means3D, device=device),
                shs=t_shs,
                colors_precomp=None,
                opacities=t_opac,
                scales=t_scales,
                rotations=t_rots,
                cov3D_precomp=None,
            )

        img_np = image.detach().cpu().numpy().transpose(2, 1, 0)

        if enable_laser:
            # 2D Crosshair drawn directly on pixels (100% reliable)
            c_x, c_y = curr_mouse[0], curr_mouse[1]
            px = int(c_x * W)
            py = int(c_y * H)

            # If it hits, the crosshair turns bright red, otherwise dark gray
            color = [1.0, 0.2, 0.2] if not math.isinf(t_hit) else [0.5, 0.5, 0.5]

            cross_radius = 15
            thickness = 2

            # Vertical line
            y_start = max(0, py - cross_radius)
            y_end = min(H, py + cross_radius)
            x_start = max(0, px - thickness)
            x_end = min(W, px + thickness)
            img_np[x_start:x_end, y_start:y_end] = color

            # Horizontal line
            x_start2 = max(0, px - cross_radius)
            x_end2 = min(W, px + cross_radius)
            y_start2 = max(0, py - thickness)
            y_end2 = min(H, py + thickness)
            img_np[x_start2:x_end2, y_start2:y_end2] = color

        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        canvas.set_image(img_np)

        # ==========================================
        # GUI OVERLAY
        # ==========================================
        gui.begin("Gaussian Effects Editor", 0.02, 0.02, 0.35, 0.35)
        gui.text("Controls:")
        gui.text(" RMB: Rotate | WASD: Move | Q/E: Zoom")
        gui.text("")

        enable_laser = gui.checkbox("Enable Laser/Flame (LMB or MMB)", enable_laser)
        laser_radius = gui.slider_float(
            "Effect Radius", laser_radius, extent * 0.01, extent * 0.2
        )
        laser_power = gui.slider_float("Beam Power", laser_power, 0.01, 1.0)

        gui.text("")
        if gui.button("Save Screenshot (HQ)"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("screenshots", exist_ok=True)
            path = f"screenshots/laser_effect_{timestamp}.png"
            imageio.imwrite(path, (img_np * 255).astype(np.uint8))
            print(f"[SCREENSHOT] Saved to {path}")

        gui.text("")
        if gui.button("RESET MODEL"):
            with torch.no_grad():
                t_shs.copy_(t_shs_orig)
            print("[INFO] Model reset to original conditions!")

        if not video_exporter.recording:
            if gui.button("Inizia Registrazione"):
                script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                video_exporter.start_recording(output_filename=f"{script_name}_{timestamp}.mp4")
        else:
            gui.text("Registrazione in corso...")
            if gui.button("Ferma Registrazione"):
                video_exporter.stop_recording()

        gui.end()

        if video_exporter.recording:
            video_frame = np.ascontiguousarray(np.flip(img_np.transpose(1, 0, 2), axis=0))
            video_exporter.add_frame(video_frame)

        window.show()


if __name__ == "__main__":
    main()
