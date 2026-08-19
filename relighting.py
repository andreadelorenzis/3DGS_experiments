import torch
import torch.nn.functional as F
import numpy as np
import taichi as ti
from plyfile import PlyData
import math
import os
import datetime
import imageio.v2 as imageio
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

ti.init(arch=ti.cuda)

from utils.ply import load_3dgs_ply
from utils.camera import get_world_to_view_matrix, getProjectionMatrix, FreeCamera
from utils.video_exporter import VideoExporter
import sys

def extract_normals_and_relight(
    means3D,
    scales,
    rots,
    base_colors,
    light_dir,
    light_color,
    light_intensity,
    campos,
    ambient=0.15,
):
    # Extracts pseudo-normals from the shortest scale axis and applies diffuse relighting
    N = means3D.shape[0]

    # 1. Convert Quaternions to Rotation Matrices
    r, i, j, k = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
    R = torch.zeros((N, 3, 3), device=means3D.device)
    R[:, 0, 0] = 1.0 - 2.0 * (j * j + k * k)
    R[:, 0, 1] = 2.0 * (i * j - r * k)
    R[:, 0, 2] = 2.0 * (i * k + r * j)
    R[:, 1, 0] = 2.0 * (i * j + r * k)
    R[:, 1, 1] = 1.0 - 2.0 * (i * i + k * k)
    R[:, 1, 2] = 2.0 * (j * k - r * i)
    R[:, 2, 0] = 2.0 * (i * k - r * j)
    R[:, 2, 1] = 2.0 * (j * k + r * i)
    R[:, 2, 2] = 1.0 - 2.0 * (i * i + j * j)

    # 2. Find the shortest axis (which approximates the surface normal for flat splats)
    shortest_axis_idx = torch.argmin(scales, dim=1)

    # 3. Extract the normal vector
    idx_expanded = shortest_axis_idx.unsqueeze(1).unsqueeze(2).expand(N, 3, 1)
    normals = torch.gather(R, 2, idx_expanded).squeeze(2)
    normals = F.normalize(normals, dim=-1)

    # 4. Flip normal to always face the camera
    view_dir = F.normalize(campos.unsqueeze(0) - means3D, dim=-1)
    dot_view = torch.sum(normals * view_dir, dim=-1, keepdim=True)
    normals = torch.where(dot_view < 0, -normals, normals)

    # 5. Calculate Diffuse Lighting (Lambertian)
    light_dir = F.normalize(light_dir, dim=-1)
    diffuse = torch.clamp(torch.sum(normals * light_dir, dim=-1, keepdim=True), 0.0, 1.0)

    # 6. Apply Lighting to Base Colors
    light_contribution = diffuse * light_intensity * light_color
    final_colors = base_colors * (ambient + light_contribution)

    return torch.clamp(final_colors, 0.0, 1.0)


# ==========================================
# 3. EDITOR AND RENDERING LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_3dgs_ply("modello.ply", max_points=10000000)
    pos_np, col_np, scale_np, rot_np, opac_np, shs_np = data["pos"], data["cols"], data["scale"], data["rot"], data["opac"], data["shs"]

    # Calculate center and true size to orbit the camera correctly
    center_np = pos_np.mean(axis=0)
    extent = float((pos_np.max(axis=0) - pos_np.min(axis=0)).max())

    t_means3D = torch.tensor(pos_np, dtype=torch.float32, device=device)
    t_base_colors = torch.tensor(col_np, dtype=torch.float32, device=device)
    t_scales = torch.tensor(scale_np, dtype=torch.float32, device=device)
    t_rots = torch.tensor(rot_np, dtype=torch.float32, device=device)
    t_opac = torch.tensor(opac_np, dtype=torch.float32, device=device).unsqueeze(1)
    t_shs = torch.tensor(shs_np, dtype=torch.float32, device=device)

    W, H = 1280, 720
    window = ti.ui.Window("3DGS Dynamic Relighting Studio", res=(W, H), show_window=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    initial_pos = center_np.copy()
    initial_pos[2] -= extent * 1.5
    camera = FreeCamera(position=initial_pos, target=center_np)

    video_exporter = VideoExporter(output_dir="results")
    auto_pan = False
    pan_angle = 0.0
    pan_radius = extent * 1.5

    # Lighting Parameters
    ambient_light = 0.15
    light_intensity = 1.5
    light_r, light_g, light_b = 1.0, 1.0, 1.0
    light_azimuth = 45.0
    light_elevation = 45.0
    show_original = False

    print("\n[EDITOR LAUNCHED]")
    print(" - LMB (Left Mouse Button) + Drag: Rotate Camera.")
    print(" - WASD: Move camera forward/left/back/right.")
    print(" - UP / DOWN: Move camera up and down.")

    while window.running:
        # Camera movement speed adapts to object scale
        speed = extent * 0.01

        if auto_pan:
            pan_angle += 0.005
            camera.pos[0] = center_np[0] + pan_radius * math.cos(pan_angle)
            camera.pos[2] = center_np[2] + pan_radius * math.sin(pan_angle)
            camera.pos[1] = center_np[1] - pan_radius * 0.2
            
            forward = center_np - camera.pos
            forward_xz = math.sqrt(forward[0]**2 + forward[2]**2)
            if forward_xz > 1e-6:
                camera.azimuth = math.atan2(forward[0], forward[2])
                camera.elevation = math.atan2(forward[1], forward_xz)
        
        cam_pos_np, target_np = camera.update(window, speed)
        cam_pos = torch.tensor(cam_pos_np, dtype=torch.float32, device=device)
        
        viewmatrix, projmatrix, t_pos, t_target, tan_fovx, tan_fovy = camera.get_matrices(
            device, W, H, znear=0.01, zfar=extent * 10.0
        )
        full_proj = viewmatrix @ projmatrix

        # Render Selection: Original vs Relighting
        if show_original:
            active_sh_degree = 3
            active_shs = t_shs
            active_colors = None
        else:
            active_sh_degree = 0
            active_shs = None

            # Compute Directional Light Vector
            l_az_rad = math.radians(light_azimuth)
            l_el_rad = math.radians(light_elevation)
            l_dir_np = np.array(
                [
                    math.cos(l_el_rad) * math.sin(l_az_rad),
                    math.sin(l_el_rad),
                    math.cos(l_el_rad) * math.cos(l_az_rad),
                ]
            )
            t_light_dir = torch.tensor(l_dir_np, dtype=torch.float32, device=device)
            t_light_color = torch.tensor(
                [light_r, light_g, light_b], dtype=torch.float32, device=device
            )

            # Apply Relighting logic
            active_colors = extract_normals_and_relight(
                t_means3D,
                t_scales,
                t_rots,
                t_base_colors,
                t_light_dir,
                t_light_color,
                light_intensity,
                cam_pos,
                ambient=ambient_light,
            )

        raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tan_fovx,
            tanfovy=tan_fovy,
            bg=torch.tensor([0.15, 0.15, 0.15], dtype=torch.float32, device=device),
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=full_proj,
            sh_degree=active_sh_degree,
            campos=cam_pos,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings)

        image, _ = rasterizer(
            means3D=t_means3D,
            means2D=torch.zeros_like(t_means3D, device=device),
            shs=active_shs,
            colors_precomp=active_colors,
            opacities=t_opac,
            scales=t_scales,
            rotations=t_rots,
            cov3D_precomp=None,
        )

        img_np = image.detach().cpu().numpy()
        img_np = img_np.transpose(2, 1, 0)
        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        canvas.set_image(img_np)

        # ==========================================
        # GRAPHICAL USER INTERFACE (GUI)
        # ==========================================
        gui.begin("Lighting Settings", 0.02, 0.02, 0.35, 0.7)

        show_original = gui.checkbox("Show Original Lighting (Baked SH)", show_original)

        if not show_original:
            gui.text("")
            gui.text("Light Intensity")
            ambient_light = gui.slider_float("Ambient Light", ambient_light, 0.0, 1.0)
            light_intensity = gui.slider_float(
                "Directional Light", light_intensity, 0.0, 5.0
            )

            gui.text("")
            gui.text("Light Color")
            light_r = gui.slider_float("Red (R)", light_r, 0.0, 1.0)
            light_g = gui.slider_float("Green (G)", light_g, 0.0, 1.0)
            light_b = gui.slider_float("Blue (B)", light_b, 0.0, 1.0)

            gui.text("")
            gui.text("Light Position")
            light_azimuth = gui.slider_float(
                "Azimuth (Horizontal)", light_azimuth, -180.0, 180.0
            )
            light_elevation = gui.slider_float(
                "Elevation (Vertical)", light_elevation, -90.0, 90.0
            )

        gui.text("")
        if gui.button("Take Photo"):
            os.makedirs("screenshots", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join("screenshots", filename)

            save_img = (img_np * 255.0).astype(np.uint8).transpose(1, 0, 2)
            save_img = np.ascontiguousarray(np.flipud(save_img))
            imageio.imwrite(filepath, save_img)
            print(f"[PHOTO] Image saved to: {filepath}")

        if not video_exporter.recording:
            if gui.button("Inizia Registrazione"):
                script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                video_exporter.start_recording(output_filename=f"{script_name}_{timestamp}.mp4")
        else:
            gui.text("Registrazione in corso...")
            if gui.button("Ferma Registrazione"):
                video_exporter.stop_recording()
                
        auto_pan_text = "Ferma Panoramica 360" if auto_pan else "Avvia Panoramica 360"
        if gui.button(auto_pan_text):
            auto_pan = not auto_pan
            if auto_pan:
                pan_radius = math.sqrt((camera.pos[0] - center_np[0])**2 + (camera.pos[2] - center_np[2])**2)
                if pan_radius < extent * 0.1:
                    pan_radius = extent * 1.5
                pan_angle = math.atan2(camera.pos[2] - center_np[2], camera.pos[0] - center_np[0])

        gui.end()

        if video_exporter.recording:
            video_frame = np.ascontiguousarray(np.flip(img_np.transpose(1, 0, 2), axis=0))
            video_exporter.add_frame(video_frame)

        window.show()


if __name__ == "__main__":
    main()
