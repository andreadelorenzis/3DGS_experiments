import taichi as ti
import numpy as np
import torch
import math

import utils.physics_engine as phys
from utils.model_loader import ModelLoader
from utils.editor import Editor
from utils.simulation import run_simulation

@ti.kernel
def draw_laser_crosshair(
    image: ti.template(), px: int, py: int, hit: int, W: int, H: int
):
    for i, j in ti.ndrange((-10, 11), (-1, 2)):
        cx, cy = px + i, py + j
        if 0 <= cx < W and 0 <= cy < H:
            if hit == 1:
                image[cx, cy, 0] = 1.0
                image[cx, cy, 1] = 0.0
                image[cx, cy, 2] = 0.0
            else:
                image[cx, cy, 0] = 1.0
                image[cx, cy, 1] = 1.0
                image[cx, cy, 2] = 1.0

    for i, j in ti.ndrange((-1, 2), (-10, 11)):
        cx, cy = px + i, py + j
        if 0 <= cx < W and 0 <= cy < H:
            if hit == 1:
                image[cx, cy, 0] = 1.0
                image[cx, cy, 1] = 0.0
                image[cx, cy, 2] = 0.0
            else:
                image[cx, cy, 0] = 1.0
                image[cx, cy, 1] = 1.0
                image[cx, cy, 2] = 1.0

class LaserEditor(Editor):
    def __init__(self, loader, initial_data):
        super().__init__(loader, initial_data)
        self.enable_laser = False
        self.laser_radius = 0.02
        self.fracture_threshold = 1.5
        
        # Laser Cutter starts simulation automatically
        self.start_simulation()
        
    def start_simulation(self):
        # Programmatically trigger the start simulation logic
        from utils.math import get_rotation_matrix, matrix_to_quaternion, q_mul
        R = get_rotation_matrix(self.rot_x, self.rot_y, self.rot_z)
        initial_pos = (
            (R @ (self.model_data["pos_local_phys"] * self.model_scale).T).T
            + self.model_data["base_offset"]
            + np.array([0.0, self.model_y_offset, 0.0], dtype=np.float32)
        )
        initial_pos = np.clip(initial_pos, 0.05, 0.85)
        phys.init_data(initial_pos)
        
        pass

    def draw_custom_gui(self, gui):
        gui.text("")
        gui.text("=== LASER CUTTER ===")
        self.enable_laser = gui.checkbox("Enable Laser Cutter (MMB to Cut)", self.enable_laser)
        self.laser_radius = gui.slider_float("Cut Thickness", self.laser_radius, 0.005, 0.05)
        self.fracture_threshold = gui.slider_float("Fracture Resistance", self.fracture_threshold, 1.1, 3.0)

    def get_fracture_threshold(self):
        return self.fracture_threshold
        
    def handle_custom_input(self, window):
        if not self.enable_laser:
            return
            
        curr_mouse = window.get_cursor_pos()
        mx, my = curr_mouse[0], curr_mouse[1]
        nx = 2.0 * mx - 1.0
        ny = 2.0 * my - 1.0
        
        W, H = 1280, 720
        fovY = 60.0 * math.pi / 180.0
        tan_fovy = math.tan(fovY / 2)
        tan_fovx = tan_fovy * (W / H)
        
        fwd_full = self.target_np - self.cam_pos_np
        fwd_full /= np.linalg.norm(fwd_full) + 1e-6
        right_full = np.cross(fwd_full, np.array([0, -1, 0]))
        right_full /= np.linalg.norm(right_full) + 1e-6
        up_full = np.cross(right_full, fwd_full)
        
        ray_dir = right_full * (nx * tan_fovx) + up_full * (ny * tan_fovy) + fwd_full
        ray_dir /= np.linalg.norm(ray_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        O = torch.tensor(self.cam_pos_np, dtype=torch.float32, device=device)
        D = torch.tensor(ray_dir, dtype=torch.float32, device=device)
        
        t_means3D_phys = phys.x.to_torch(device=device).to(torch.float32)
        t_active_phys = phys.active.to_torch(device=device).to(torch.bool)
        
        PO = t_means3D_phys - O
        t_along_ray = torch.sum(PO * D, dim=1)
        proj = t_along_ray.unsqueeze(1) * D
        dist = torch.norm(PO - proj, dim=1)
        
        # Ray must be close to the point, pointing FORWARD, and the point must be ACTIVE
        hit_mask = (dist < self.laser_radius) & (t_along_ray > 0) & t_active_phys
        is_hit = hit_mask.any().item()
        
        if is_hit and window.is_pressed(ti.ui.MMB):
            t_active_phys[hit_mask] = False
            # Zero-copy push back to Taichi GPU
            phys.active.from_numpy(t_active_phys.cpu().numpy().astype(np.int32))
            
        self.last_hit = is_hit
        self.last_mouse = curr_mouse
        
    def post_render_hook(self, render_buffer, window):
        if not self.enable_laser:
            return
        if not hasattr(self, 'last_hit'):
            return
            
        W, H = 1280, 720
        c_x, c_y = self.last_mouse[0], self.last_mouse[1]
        px = int(c_x * W)
        py = int(c_y * H)
        hit_val = 1 if self.last_hit else 0
        
        draw_laser_crosshair(render_buffer, px, py, hit_val, W, H)

def main():
    print("[INFO] Starting PhysGaussian Studio (Laser Cutter Mode)...")
    loader = ModelLoader(
        max_render_points=phys.MAX_RENDER_POINTS, 
        max_phys_points=phys.MAX_PHYS_POINTS,
        use_mesh=False
    )
    loader._load_worker("modello.ply")
    init_data = loader.get_loaded_data()
    
    editor = LaserEditor(loader, init_data)
    trigger_simulation, sim_params = editor.run()

    if trigger_simulation and sim_params is not None:
        run_simulation(*sim_params)
        print("\nOperation completed! Video is ready.")

if __name__ == "__main__":
    main()
