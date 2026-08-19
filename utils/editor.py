import time
import math
import torch
import numpy as np
import taichi as ti
import os
import sys

try:
    import psutil
except ImportError:
    psutil = None

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.camera import get_world_to_view_matrix, getProjectionMatrix, FreeCamera
from utils.math import get_rotation_matrix, matrix_to_quaternion, q_mul
from utils.physics import compute_cov3d_deformed
import utils.physics_engine as phys
from utils.video_exporter import VideoExporter

class Editor:
    def __init__(self, loader, initial_data):
        self.loader = loader
        self.update_model_data(initial_data)

    def update_model_data(self, new_data):
        self.model_data = new_data
        self.N = new_data["N"]
        self.M = new_data["M"]
        phys.active_N[None] = self.N
        phys.active_N_total[None] = self.N
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.model_y_offset = 0.0
        self.model_scale = 0.5
        
        self.mesh_indices = None
        if "mesh_f_np" in new_data and new_data["mesh_f_np"] is not None:
            f_np = new_data["mesh_f_np"].flatten()
            self.mesh_indices = ti.field(dtype=int, shape=f_np.size)
            self.mesh_indices.from_numpy(f_np)

    def run(self):
        print("[INFO] Warming up Taichi kernels to prevent UI freezing...")
        dummy_pos = np.zeros((phys.MAX_PHYS_POINTS, 3), dtype=np.float32)
        phys.init_data(dummy_pos)
        phys.mpm_substep(5000.0, 5000.0, 9.81, 0, 5.0, 30.0, 0.05, 0.7, 1.0, 1e6)
        print("[INFO] Warmup complete.")

        window = ti.ui.Window("PhysGaussian Studio - Editor & Preview", res=(1280, 720), show_window=True)
        canvas = window.get_canvas()
        gui = window.get_gui()
        scene = window.get_scene()
        camera = ti.ui.Camera()

        self.video_exporter = VideoExporter(output_dir="results")

        # GUI and simulation state initialization
        self.non_gui_E = 5000.0
        self.non_gui_grav = 9.81
        self.material_idx = phys.MATERIAL_ELASTIC
        self.hardening = 5.0
        self.friction_deg = 30.0
        self.yield_stress = 0.05
        self.floor_friction = 0.7

        yield_stress = 0.05
        floor_friction = 0.7

        self.show_only_lowpoly = False

        self.cam_pos_np = np.array([0.5, 0.4, 1.3], dtype=np.float32)
        self.target_np = np.array([0.5, 0.4, 0.3], dtype=np.float32)
        self.free_camera = FreeCamera(position=self.cam_pos_np, target=self.target_np)

        self.show_max_quality = getattr(self, "show_max_quality", False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        t_colors = torch.tensor(self.model_data["render_cols_np"], dtype=torch.float32, device=device)
        t_shs = torch.tensor(self.model_data["render_shs_np"], dtype=torch.float32, device=device)
        t_opac = torch.tensor(self.model_data["render_opac_np"], dtype=torch.float32, device=device).unsqueeze(1)

        self.enable_crusher = False
        self.crusher_mass = 50.0

        # Floor Gaussians
        num_floor = len(phys.floor_np)
        t_floor_pos = torch.tensor(phys.floor_np, dtype=torch.float32, device=device)
        t_floor_cols = torch.full((num_floor, 3), 0.5, dtype=torch.float32, device=device)
        t_floor_opac = torch.full((num_floor, 1), 1.0, dtype=torch.float32, device=device)
        t_floor_shs = torch.zeros((num_floor, 16, 3), dtype=torch.float32, device=device)
        t_floor_scale = torch.tensor([0.02, 0.001, 0.02], dtype=torch.float32, device=device).unsqueeze(0).repeat(num_floor, 1)
        
        t_crusher_shs = torch.zeros((phys.CRUSHER_POINTS, 16, 3), dtype=torch.float32, device=device)
        C0 = 0.28209479177387814
        t_crusher_shs[:, 0, 0] = (1.0 - 0.5) / C0
        t_crusher_shs[:, 0, 1] = (0.2 - 0.5) / C0
        t_crusher_shs[:, 0, 2] = (0.2 - 0.5) / C0
        t_crusher_opac = torch.full((phys.CRUSHER_POINTS, 1), 99.0, dtype=torch.float32, device=device)
        t_crusher_scale = torch.tensor([0.003, 0.003, 0.003], dtype=torch.float32, device=device).unsqueeze(0).repeat(phys.CRUSHER_POINTS, 1)
        t_floor_rot = torch.zeros((num_floor, 4), dtype=torch.float32, device=device)
        t_floor_rot[:, 0] = 1.0
        t_floor_cov3d = compute_cov3d_deformed(
            t_floor_scale, t_floor_rot, torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_floor, 1, 1)
        )

        # Crusher Gaussians
        t_crusher_rot = torch.zeros((phys.CRUSHER_POINTS, 4), dtype=torch.float32, device=device)
        t_crusher_rot[:, 0] = 1.0
        t_crusher_cov3d = compute_cov3d_deformed(
            t_crusher_scale, t_crusher_rot, torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(phys.CRUSHER_POINTS, 1, 1)
        )

        crusher_pts_np = np.zeros((phys.CRUSHER_POINTS, 3), dtype=np.float32)
        for i in range(10):
            for k in range(10):
                idx = i * 10 + k
                crusher_pts_np[idx] = [i / 9.0 * 0.8 + 0.1, 0.0, k / 9.0 * 0.8 + 0.1]
                
        t_crusher_base_pos = torch.zeros((phys.CRUSHER_POINTS, 3), dtype=torch.float32, device=device)
        for i in range(10):
            for k in range(10):
                idx = i * 10 + k
                t_crusher_base_pos[idx] = torch.tensor([i / 9.0 * 0.8 + 0.1, 0.0, k / 9.0 * 0.8 + 0.1], device=device)
        crusher_field = ti.Vector.field(3, dtype=float, shape=phys.CRUSHER_POINTS)

        last_preview_params = None
        t_means3D = None
        t_scale = None
        t_rots = None

        is_simulating = False
        last_frame_time = time.time()
        sim_time_accumulator = 0.0
        fps_ema = 60.0
        rtf_ema = 1.0
        phys_ms = 0.0
        raster_ms = 0.0
        skin_ms = 0.0
        num_substeps = 0
        collect_metrics = False
        metrics_frames = 0
        metrics_acc = {"fps": 0.0, "rtf": 0.0, "phys": 0.0, "skin": 0.0, "raster": 0.0}
        render_percentage = 1.0
        gui_max_substeps = 80
        safe_cfl_limit = (phys.dx / math.sqrt(20000.0 / phys.p_rho)) * 0.8
        dynamic_dt = 2.5e-4
        shuffle_idx = torch.randperm(self.M, device=device)
        render_buffer = ti.field(dtype=ti.f32, shape=(1280, 720, 3))

        t_local_offset_sim = None
        t_skin_idx_sim = None
        t_rot_sim = None
        t_scale_sim = None

        print("\nEDITOR ACTIVE!")
        print(" - WASD: Move camera forward/backward/sideways.")
        print(" - UP / DOWN Arrows: Vertical translation.")
        print(" - Left Mouse Button (LMB) + Drag: Rotate camera around the object.")
        print(" - Left Panel: Adjust physics, Rotation, Scale, and Model Height.")
        print(" - Button: Start Simulation & Export.\n")
        print(f"[INFO] Editor/physics: {self.N} particles | Final video: {self.M} gaussians.\n")

        trigger_simulation = False
        sim_params = None
        loading_indicator = ti.Vector.field(2, dtype=ti.f32, shape=1)

        while window.running:
            if self.loader.is_loading:
                loading_indicator[0] = [0.5 + 0.05 * math.cos(time.time() * 5), 0.5 + 0.05 * math.sin(time.time() * 5)]
                canvas.circles(loading_indicator, radius=0.02, color=(1.0, 0.6, 0.1))
                gui.begin("Loading", 0.4, 0.4, 0.2, 0.1)
                gui.text("Loading new model, please wait...")
                gui.end()
                window.show()
                continue

            self.handle_custom_input(window)

            new_data = self.loader.get_loaded_data()
            if new_data is not None:
                self.update_model_data(new_data)
                t_colors = torch.tensor(self.model_data["render_cols_np"], dtype=torch.float32, device=device)
                t_shs = torch.tensor(self.model_data["render_shs_np"], dtype=torch.float32, device=device)
                t_opac = torch.tensor(self.model_data["render_opac_np"], dtype=torch.float32, device=device).unsqueeze(1)
                is_simulating = False
                last_preview_params = None
                shuffle_idx = torch.randperm(self.M, device=device)
                print(f"[INFO] New model loaded! Editor/physics: {self.N} particles | Final video: {self.M} gaussians.\n")

            curr_time = time.time()
            dt_real = curr_time - last_frame_time
            last_frame_time = curr_time
            if dt_real > 0:
                fps_ema = fps_ema * 0.9 + (1.0 / dt_real) * 0.1

            speed = 0.005
            self.cam_pos_np, self.target_np = self.free_camera.update(window, speed)
            camera.position(self.cam_pos_np[0], self.cam_pos_np[1], self.cam_pos_np[2])
            camera.lookat(self.target_np[0], self.target_np[1], self.target_np[2])
            camera.up(0, -1, 0)
            scene.set_camera(camera)

            if is_simulating:
                if self.override_physics_step(dt_real):
                    pass
                else:
                    sim_time_accumulator += dt_real
                    mu = self.non_gui_E / (2 * (1 + phys.nu))
                    lam = self.non_gui_E * phys.nu / ((1 + phys.nu) * (1 - 2 * phys.nu))
                    num_substeps = 0
                    t0_phys = time.time()
                    while sim_time_accumulator > 0 and num_substeps < gui_max_substeps:
                        if self.enable_crusher:
                            if phys.crusher_y_ti[None] < 0.6:
                                phys.crusher_y_ti[None] += 0.5 * phys.dt
                        phys.mpm_substep(
                            mu, lam, self.non_gui_grav, self.material_idx, self.hardening, self.friction_deg, self.yield_stress, self.floor_friction,
                            self.crusher_mass if self.enable_crusher else 1.0, self.get_fracture_threshold()
                        )
                        sim_time_accumulator -= dynamic_dt
                        num_substeps += 1
                    if num_substeps == gui_max_substeps:
                        sim_time_accumulator = 0.0
                    phys_ms = (time.time() - t0_phys) * 1000.0
                    if dt_real > 0:
                        rtf_ema = rtf_ema * 0.9 + (num_substeps * dynamic_dt / dt_real) * 0.1

                t0_skin = time.time()
                t_means3D_phys = phys.x.to_torch(device=device).to(torch.float32)
                F_tensor = phys.F.to_torch(device=device).to(torch.float32)
                current_M = max(1, int(self.M * render_percentage))
                with torch.no_grad():
                    active_render_indices = shuffle_idx[:current_M]
                    active_bind_idx = t_skin_idx_sim[active_render_indices]
                    
                    t_active_phys = phys.active.to_torch(device=device).to(torch.bool)
                    cut_mask = ~t_active_phys[active_bind_idx]
                    t_current_opac = t_opac[active_render_indices].clone()
                    t_current_opac[cut_mask] = 0.0

                    rotated_offset = torch.bmm(
                        F_tensor[active_bind_idx],
                        t_local_offset_sim[active_render_indices].unsqueeze(2),
                    ).squeeze(2)
                    t_means3D_render = t_means3D_phys[active_bind_idx] + rotated_offset
                    cov3D_precomp = compute_cov3d_deformed(
                        t_scale_sim[active_render_indices],
                        t_rot_sim[active_render_indices],
                        F_tensor[active_bind_idx],
                    )
                skin_ms = (time.time() - t0_skin) * 1000.0

                t0_raster = time.time()
                # FAKE SCALE UP to bypass 3DGS anti-aliasing blur
                render_scale_up = float(10.0 / self.model_data.get("extent", 1.0))
                
                W, H = 1280, 720
                viewmatrix, projmatrix, t_cam_pos, t_target_tch, tan_fovx, tan_fovy = self.free_camera.get_matrices(
                    device, W, H, znear=0.01 * render_scale_up, zfar=100.0 * render_scale_up
                )
                
                t_cam_pos_scaled = t_cam_pos * render_scale_up
                viewmatrix_scaled = get_world_to_view_matrix(t_cam_pos_scaled, t_target_tch * render_scale_up, device)
                full_proj_scaled = viewmatrix_scaled @ projmatrix
                
                raster_settings = GaussianRasterizationSettings(
                    image_height=H, image_width=W, tanfovx=tan_fovx, tanfovy=tan_fovy,
                    bg=torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device),
                    scale_modifier=1.0, viewmatrix=viewmatrix_scaled, projmatrix=full_proj_scaled,
                    sh_degree=3, campos=t_cam_pos_scaled, prefiltered=False, debug=False,
                )
                rasterizer = GaussianRasterizer(raster_settings)
                with torch.no_grad():
                    if self.enable_crusher:
                        t_crusher_pos = t_crusher_base_pos.clone()
                        t_crusher_pos[:, 1] = phys.crusher_y_ti[None]
                        t_render_pos = torch.cat([t_means3D_render, t_floor_pos, t_crusher_pos], dim=0)
                        t_render_shs = torch.cat([t_shs[active_render_indices], t_floor_shs, t_crusher_shs], dim=0)
                        t_render_opac = torch.cat([t_current_opac, t_floor_opac, t_crusher_opac], dim=0)
                        t_render_cov3d = torch.cat([cov3D_precomp, t_floor_cov3d, t_crusher_cov3d], dim=0)
                    else:
                        t_render_pos = torch.cat([t_means3D_render, t_floor_pos], dim=0)
                        t_render_shs = torch.cat([t_shs[active_render_indices], t_floor_shs], dim=0)
                        t_render_opac = torch.cat([t_current_opac, t_floor_opac], dim=0)
                        t_render_cov3d = torch.cat([cov3D_precomp, t_floor_cov3d], dim=0)

                    t_render_pos_scaled = t_render_pos * render_scale_up

                    image, _ = rasterizer(
                        means3D=t_render_pos_scaled, means2D=torch.zeros((t_render_pos.shape[0], 2), device=device),
                        shs=t_render_shs, colors_precomp=None, opacities=t_render_opac,
                        scales=None, rotations=None, cov3D_precomp=t_render_cov3d * (render_scale_up ** 2),
                    )
                image_pt = image.detach().permute(2, 1, 0).contiguous()
                image_pt = torch.clamp(image_pt, 0.0, 1.0)
                render_buffer.from_torch(image_pt)
                
                self.post_render_hook(render_buffer, window)
                
                canvas.set_image(render_buffer)
                raster_ms = (time.time() - t0_raster) * 1000.0

            else:
                R_preview = get_rotation_matrix(self.rot_x, self.rot_y, self.rot_z)
                live_pos = (
                    (R_preview @ (self.model_data["pos_local_phys"] * self.model_scale).T).T
                    + self.model_data["base_offset"]
                    + np.array([0.0, self.model_y_offset, 0.0], dtype=np.float32)
                )
                phys.x.from_numpy(live_pos)

                if self.show_max_quality:
                    current_preview_params = (self.rot_x, self.rot_y, self.rot_z, self.model_scale, self.model_y_offset)
                    if current_preview_params != last_preview_params:
                        dense_pos = (
                            (R_preview @ (self.model_data["render_pos_local"] * self.model_scale).T).T
                            + self.model_data["base_offset"]
                            + np.array([0.0, self.model_y_offset, 0.0], dtype=np.float32)
                        )
                        t_means3D = torch.tensor(dense_pos, dtype=torch.float32, device=device)
                        t_scale = torch.tensor(self.model_data["render_scale_np"] * self.model_scale, dtype=torch.float32, device=device)
                        model_q = matrix_to_quaternion(R_preview)
                        rotated_rot = q_mul(model_q, self.model_data["render_rot_np"])
                        t_rots = torch.tensor(rotated_rot, dtype=torch.float32, device=device)
                        last_preview_params = current_preview_params

                    W, H = 1280, 720
                    fovY = 60.0 * math.pi / 180.0
                    tan_fovy = math.tan(fovY / 2)
                    tan_fovx = tan_fovy * (W / H)
                    fovX = 2.0 * math.atan(tan_fovx)

                    t_cam_pos = torch.tensor(self.cam_pos_np, dtype=torch.float32, device=device)
                    t_target = torch.tensor(self.target_np, dtype=torch.float32, device=device)
                    
                    # Use the actual model extent to bypass 3DGS anti-aliasing blur exactly
                    render_scale_up = float(10.0 / self.model_data.get("extent", 1.0))
                    
                    viewmatrix = get_world_to_view_matrix(t_cam_pos, t_target, device)
                    projmatrix = getProjectionMatrix(0.01 * render_scale_up, 100.0 * render_scale_up, fovX, fovY, device)

                    t_cam_pos_scaled = t_cam_pos * render_scale_up
                    
                    # Compute scaled view and projection matrices
                    viewmatrix_scaled = get_world_to_view_matrix(t_cam_pos_scaled, t_target * render_scale_up, device)
                    full_proj_scaled = viewmatrix_scaled @ projmatrix
                    
                    raster_settings = GaussianRasterizationSettings(
                        image_height=H, image_width=W, tanfovx=tan_fovx, tanfovy=tan_fovy,
                        bg=torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device),
                        scale_modifier=1.0, viewmatrix=viewmatrix_scaled, projmatrix=full_proj_scaled,
                        sh_degree=3, campos=t_cam_pos_scaled, prefiltered=False, debug=False,
                    )
                    rasterizer = GaussianRasterizer(raster_settings)

                    if self.enable_crusher:
                        current_crusher_y = max(np.min(dense_pos[:, 1]) - 0.05, 0.05)
                        t_crusher_pos = t_crusher_base_pos.clone()
                        t_crusher_pos[:, 1] = current_crusher_y
                        
                        t_render_pos = torch.cat([t_means3D, t_floor_pos, t_crusher_pos], dim=0)
                        t_render_shs = torch.cat([t_shs, t_floor_shs, t_crusher_shs], dim=0)
                        t_render_opac = torch.cat([t_opac, t_floor_opac, t_crusher_opac], dim=0)
                        t_render_scales = torch.cat([t_scale, t_floor_scale, t_crusher_scale], dim=0)
                        t_render_rots = torch.cat([t_rots, t_floor_rot, t_crusher_rot], dim=0)
                    else:
                        t_render_pos = torch.cat([t_means3D, t_floor_pos], dim=0)
                        t_render_shs = torch.cat([t_shs, t_floor_shs], dim=0)
                        t_render_opac = torch.cat([t_opac, t_floor_opac], dim=0)
                        t_render_scales = torch.cat([t_scale, t_floor_scale], dim=0)
                        t_render_rots = torch.cat([t_rots, t_floor_rot], dim=0)

                    t_render_pos_scaled = t_render_pos * render_scale_up

                    image, _ = rasterizer(
                        means3D=t_render_pos_scaled, means2D=torch.zeros((t_render_pos.shape[0], 2), device=device),
                        shs=t_render_shs, colors_precomp=None, opacities=t_render_opac,
                        scales=t_render_scales * render_scale_up, rotations=t_render_rots, cov3D_precomp=None,
                    )

                    img_np = image.detach().cpu().numpy()
                    img_np = img_np.transpose(2, 1, 0)
                    img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
                    canvas.set_image(img_np)
                else:
                    scene.ambient_light((0.3, 0.3, 0.3))
                    scene.point_light(pos=(1, 2, -1), color=(1, 1, 1))
                    if self.show_only_lowpoly and self.mesh_indices is not None:
                        scene.mesh(phys.x, indices=self.mesh_indices, color=(0.75, 0.3, 0.3), two_sided=True)
                    else:
                        if self.mesh_indices is not None:
                            scene.mesh(phys.x, indices=self.mesh_indices, color=(0.7, 0.7, 0.7), two_sided=True)
                            scene.particles(phys.x, radius=0.003, color=(0.2, 0.6, 0.9))
                        else:
                            scene.particles(phys.x, radius=0.003, color=(0.2, 0.6, 0.9))

                    scene.particles(phys.floor_field, radius=0.02, color=(0.3, 0.3, 0.3))
                    if self.enable_crusher:
                        current_crusher_y = max(np.min(live_pos[:, 1]) - 0.05, 0.05)
                        c_np = crusher_pts_np.copy()
                        c_np[:, 1] = current_crusher_y
                        crusher_field.from_numpy(c_np)
                        scene.particles(crusher_field, radius=0.008, color=(0.8, 0.1, 0.1))
                    canvas.scene(scene)

            # Draw GUI Panels
            gui.begin("PhysGaussian Control Panel", 0.02, 0.02, 0.32, 0.9)

            gui.text("Material")
            if gui.button(("> " if self.material_idx == phys.MATERIAL_ELASTIC else "   ") + "Elastic"): self.material_idx = phys.MATERIAL_ELASTIC
            if gui.button(("> " if self.material_idx == phys.MATERIAL_METAL else "   ") + "Metal (Plastic)"): self.material_idx = phys.MATERIAL_METAL
            if gui.button(("> " if self.material_idx == phys.MATERIAL_FOAM else "   ") + "Foam / Gel"): self.material_idx = phys.MATERIAL_FOAM
            if gui.button(("> " if self.material_idx == phys.MATERIAL_SAND else "   ") + "Granular (sand)"): self.material_idx = phys.MATERIAL_SAND
            gui.text(f"Selected: {phys.MATERIAL_NAMES[self.material_idx]}")

            gui.text("")
            gui.text("MPM Physics Parameters")
            self.non_gui_E = gui.slider_float("Young's Modulus (E)", self.non_gui_E, 500.0, 20000.0)
            self.non_gui_grav = gui.slider_float("Gravity", self.non_gui_grav, 0.0, 20.0)
            self.floor_friction = gui.slider_float("Floor Friction", self.floor_friction, 0.0, 1.0)

            if self.material_idx == phys.MATERIAL_ELASTIC: pass
            elif self.material_idx == phys.MATERIAL_METAL: self.hardening = gui.slider_float("Hardening (metal)", self.hardening, 0.0, 15.0)
            elif self.material_idx == phys.MATERIAL_FOAM: self.yield_stress = gui.slider_float("Yield stress (foam/gel)", self.yield_stress, 0.001, 0.3)
            elif self.material_idx == phys.MATERIAL_SAND: self.friction_deg = gui.slider_float("Friction angle (sand)", self.friction_deg, 5.0, 60.0)

            gui.text("")
            gui.text("Model Transformation")
            self.model_y_offset = gui.slider_float("Height (Y Axis)", self.model_y_offset, -0.4, 0.4)
            self.rot_x = gui.slider_float("Rotation X (deg)", self.rot_x, 0.0, 360.0)
            self.rot_y = gui.slider_float("Rotation Y (deg)", self.rot_y, 0.0, 360.0)
            self.rot_z = gui.slider_float("Rotation Z (deg)", self.rot_z, 0.0, 360.0)

            gui.text("")
            self.show_max_quality = gui.checkbox("Dense 3DGS Preview (Editor)", self.show_max_quality)
            if self.mesh_indices is not None:
                self.show_only_lowpoly = gui.checkbox("View ONLY Low-Poly Mesh", self.show_only_lowpoly)
            self.enable_crusher = gui.checkbox("Enable Crusher (Heavy Square)", self.enable_crusher)
            if self.enable_crusher: self.crusher_mass = gui.slider_float("Crusher Mass", self.crusher_mass, 1.0, 100.0)

            gui.text("")
            if not self.video_exporter.recording:
                if gui.button("Inizia Registrazione"):
                    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    self.video_exporter.start_recording(output_filename=f"{script_name}_{timestamp}.mp4")
            else:
                gui.text("Registrazione in corso...")
                if gui.button("Ferma Registrazione"):
                    self.video_exporter.stop_recording()

            gui.text("")
            gui.text("=== MODEL SELECTION ===")
            for mod in self.loader.list_models():
                if gui.button(mod):
                    if not self.loader.is_loading:
                        self.loader.async_load_model(mod)

            gui.text("")
            if not is_simulating:
                if gui.button("Start Simulation (Real-Time)"):
                    R = get_rotation_matrix(self.rot_x, self.rot_y, self.rot_z)
                    initial_pos = (
                        (R @ (self.model_data["pos_local_phys"] * self.model_scale).T).T
                        + self.model_data["base_offset"]
                        + np.array([0.0, self.model_y_offset, 0.0], dtype=np.float32)
                    )
                    
                    if self.enable_crusher:
                        phys.enable_crusher_ti[None] = 1
                        phys.crusher_y_ti[None] = max(np.min(initial_pos[:, 1]) - 0.05, 0.05)
                    else:
                        phys.enable_crusher_ti[None] = 0
                    
                    initial_pos = np.clip(initial_pos, 0.05, 0.85)
                    phys.init_data(initial_pos)

                    local_offset_world_np = (R @ (self.model_data["local_offset_canonical_np"] * self.model_scale).T).T
                    t_local_offset_sim = torch.tensor(local_offset_world_np, dtype=torch.float32, device=device)
                    t_skin_idx_sim = torch.tensor(self.model_data["skin_idx_np"], dtype=torch.long, device=device)
                    t_scale_sim = torch.tensor(self.model_data["render_scale_np"] * self.model_scale, dtype=torch.float32, device=device)
                    model_q = matrix_to_quaternion(R)
                    rotated_rot = q_mul(model_q, self.model_data["render_rot_np"])
                    t_rot_sim = torch.tensor(rotated_rot, dtype=torch.float32, device=device)
                    is_simulating = True

                if gui.button("Start Simulation & Export to MP4"):
                    sim_params = (
                        self.model_data,
                        self.non_gui_E,
                        self.non_gui_grav,
                        self.rot_x,
                        self.rot_y,
                        self.rot_z,
                        0.5,
                        self.model_y_offset,
                        torch.tensor(self.cam_pos_np, dtype=torch.float32, device="cuda"),
                        torch.tensor(self.target_np, dtype=torch.float32, device="cuda"),
                        self.material_idx,
                        self.hardening,
                        self.friction_deg,
                        self.yield_stress,
                        self.floor_friction,
                        self.crusher_mass if self.enable_crusher else 1.0,
                        self.enable_crusher,
                        self.get_fracture_threshold(),
                    )
                    trigger_simulation = True
                    gui.end()
                    break
            else:
                if gui.button("Stop Simulation (Back to Editor)"):
                    is_simulating = False

                gui.text("")
                gui.text("=== REAL-TIME BALANCING ===")
                render_percentage = gui.slider_float("Render Detail (%)", render_percentage, 0.05, 1.0)
                gui_max_substeps = gui.slider_int("Max Substeps / Frame", gui_max_substeps, 10, 500)
                cfl_max_disp = float(safe_cfl_limit * 1e4)
                dt_slider_val = gui.slider_float(f"Time-Step (x10^-4) [Max: {cfl_max_disp:.2f}]", dynamic_dt * 1e4, 0.5, cfl_max_disp)
                dynamic_dt = dt_slider_val * 1e-4

                gui.text("")
                gui.text("=== PROFILER & METRICS ===")
                gui.text(f"Graphics FPS: {fps_ema:.1f}")
                gui.text(f"Real-Time Factor: {rtf_ema:.2f}x")
                gui.text(f"Physics Steps: {num_substeps}")
                gui.text(f"Physics Time: {phys_ms:.1f} ms")
                gui.text(f"Skinning Time: {skin_ms:.1f} ms")
                gui.text(f"Raster Time: {raster_ms:.1f} ms")

                mem_info = "N/A"
                ram_mb, vram_mb = 0, 0
                if psutil:
                    ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    vram_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
                    mem_info = f"RAM: {ram_mb:.0f} MB | VRAM: {vram_mb:.0f} MB"
                gui.text(mem_info)

                if collect_metrics:
                    metrics_frames += 1
                    metrics_acc["fps"] += fps_ema
                    metrics_acc["rtf"] += rtf_ema
                    metrics_acc["phys"] += phys_ms
                    metrics_acc["skin"] += skin_ms
                    metrics_acc["raster"] += raster_ms
                    gui.text(f"Collecting Metrics... {metrics_frames}/60")
                    if metrics_frames >= 60:
                        collect_metrics = False
                        res = f"Metrics (60-frame average):\nFPS: {metrics_acc['fps']/60:.1f}\nRTF: {metrics_acc['rtf']/60:.2f}x\nPhysics: {metrics_acc['phys']/60:.1f}ms\nSkinning: {metrics_acc['skin']/60:.1f}ms\nRaster: {metrics_acc['raster']/60:.1f}ms\nRAM: {ram_mb:.0f}MB\nVRAM: {vram_mb:.0f}MB"
                        try:
                            import tkinter as tk
                            root = tk.Tk()
                            root.withdraw()
                            root.clipboard_clear()
                            root.clipboard_append(res)
                            root.update()
                            root.destroy()
                            print("[INFO] Metrics copied to clipboard!")
                        except:
                            print("[ERROR] Tkinter failed, printing to console:\n", res)
                else:
                    if gui.button("Copy Metrics (2s Average)"):
                        collect_metrics = True
                        metrics_frames = 0
                        for k in metrics_acc:
                            metrics_acc[k] = 0.0

            self.draw_custom_gui(gui)
            gui.end()
            
            if getattr(self, 'video_exporter', None) is not None and self.video_exporter.recording:
                if is_simulating:
                    img_window = render_buffer.to_numpy()
                else:
                    img_window = window.get_image_buffer_as_numpy()[:, :, :3]
                    
                video_frame = np.ascontiguousarray(np.flip(img_window.transpose(1, 0, 2), axis=0))
                if video_frame.dtype == np.float32 or video_frame.dtype == np.float64:
                    video_frame = (np.clip(video_frame, 0.0, 1.0) * 255.0).astype(np.uint8)
                self.video_exporter.add_frame(video_frame)
                
            window.show()

        del window
        return trigger_simulation, sim_params

    # Extensibility Hooks for subclasses (LaserEditor, RigidBodyEditor, etc.)
    def handle_custom_input(self, window):
        pass

    def draw_custom_gui(self, gui):
        pass

    def get_fracture_threshold(self):
        return 1e6

    def override_physics_step(self, dt_real):
        # Return True if a custom physics step (like Rigid Body) is performed.
        return False
        
    def post_render_hook(self, render_buffer, window):
        pass
