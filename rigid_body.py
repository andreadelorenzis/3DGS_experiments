import torch
import numpy as np
import taichi as ti
import open3d as o3d
import time

from utils.math import quat_mult, quat_to_matrix
from utils.model_loader import ModelLoader
from utils.editor import Editor
import utils.physics_engine as phys

class RigidBody:
    def __init__(self, proxy_vertices_np, start_pos, mass=1.0):
        self.mass = mass
        self.proxy_local = torch.tensor(proxy_vertices_np, dtype=torch.float32)

        # Using a sphere's inertia tensor for simplicity
        self.I_local = torch.eye(3, dtype=torch.float32) * (mass * 0.1)
        self.I_inv_local = torch.inverse(self.I_local)

        self.pos = torch.tensor(start_pos, dtype=torch.float32)
        self.vel = torch.zeros(3, dtype=torch.float32)

        self.quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        self.ang_vel = torch.zeros(3, dtype=torch.float32)
        self.sleep_counter = 0

    def step(self, dt, gravity=9.8, restitution=0.5, floor_y=0.9):
        # 0. Rest state (Sleep) detection to eliminate micro-vibrations
        speed = torch.norm(self.vel)
        ang_speed = torch.norm(self.ang_vel)
        R = quat_to_matrix(self.quat)
        world_proxy = self.proxy_local @ R.T + self.pos
        max_y, max_idx = torch.max(world_proxy[:, 1], dim=0)

        is_touching = max_y >= floor_y - 0.005
        if is_touching and speed < 0.1 and ang_speed < 0.1:
            self.sleep_counter += 1
        else:
            self.sleep_counter = 0

        if self.sleep_counter > 20:
            self.vel *= 0.0
            self.ang_vel *= 0.0
            if max_y > floor_y:
                self.pos[1] -= max_y - floor_y 
            return

        # 1. Integrate forces
        self.vel[1] += gravity * dt

        # 2. Integrate position and rotation
        self.pos += self.vel * dt

        omega_q = torch.tensor([0.0, self.ang_vel[0], self.ang_vel[1], self.ang_vel[2]], dtype=torch.float32)
        dq = 0.5 * quat_mult(omega_q.unsqueeze(0), self.quat.unsqueeze(0)).squeeze(0)
        self.quat += dq * dt
        self.quat = self.quat / torch.norm(self.quat)

        # Continuous damping
        self.vel *= 0.98
        self.ang_vel *= 0.98

        # 3. Collision Detection
        R = quat_to_matrix(self.quat)
        world_proxy = self.proxy_local @ R.T + self.pos
        max_y, max_idx = torch.max(world_proxy[:, 1], dim=0)

        if max_y < floor_y:
            return  

        r = world_proxy[max_idx] - self.pos
        v_p = self.vel + torch.cross(self.ang_vel, r, dim=0)

        if v_p[1] < 0:
            return

        n = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
        I_world = R @ self.I_local @ R.T
        I_world_inv = R @ self.I_inv_local @ R.T

        r_cross_n = torch.cross(r, n, dim=0)
        term1 = 1.0 / self.mass
        term2 = torch.dot(r_cross_n, I_world_inv @ r_cross_n)
        v_n = torch.dot(v_p, n)

        if v_n > -0.3:
            restitution_eff = 0.0
        else:
            restitution_eff = restitution

        j = -(1.0 + restitution_eff) * v_n / (term1 + term2)
        impulse = j * n
        self.vel += impulse / self.mass
        self.ang_vel += I_world_inv @ torch.cross(r, impulse, dim=0)

        self.vel[0] *= 0.5
        self.vel[2] *= 0.5

        penetration = max_y - floor_y
        if penetration > 0:
            self.pos[1] -= penetration * 0.8

@ti.kernel
def update_phys_from_rigid_body(
    pos_local_field: ti.template(),
    pos_world: ti.math.vec3,
    R_mat: ti.math.mat3,
    N: int
):
    for i in range(N):
        # We explicitly update the MPM buffers to fake an MPM simulation!
        phys.x[i] = pos_world + R_mat @ pos_local_field[i]
        phys.F[i] = R_mat

class RigidBodyEditor(Editor):
    def __init__(self, loader, initial_data):
        super().__init__(loader, initial_data)
        
        # UI variables
        self.pause_physics = False
        self.gravity = 9.8
        self.restitution = 0.3
        self.dt_phys = 1.0 / 120.0
        self.sim_time = 0.0
        
        # Setup Rigid Body Collision Mesh
        print("Generating Proxy Mesh (Convex Hull)...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.model_data["pos_local_phys"])
        hull, _ = pcd.compute_convex_hull()
        proxy_vertices = np.asarray(hull.vertices)
        
        self.rb = RigidBody(proxy_vertices, [0.5, 0.4, 0.5])
        
        # Upload local positions to Taichi to efficiently update phys.x in the callback
        self.local_pos_ti = ti.Vector.field(3, dtype=float, shape=self.model_data["N"])
        self.local_pos_ti.from_numpy(self.model_data["pos_local_phys"])

        # Override transformation defaults
        self.model_y_offset = -0.1
        self.model_scale = 1.0 
        self.show_max_quality = True

    def draw_custom_gui(self, gui):
        gui.text("")
        gui.text("=== RIGID BODY PHYSICS ===")
        self.pause_physics = gui.checkbox("Physics Pause", self.pause_physics)
        self.gravity = gui.slider_float("Gravity", self.gravity, 0.0, 30.0)
        self.restitution = gui.slider_float("Bounce (Restitution)", self.restitution, 0.0, 1.0)
        
        if gui.button("Throw in the air!"):
            self.rb.pos = torch.tensor([0.5, 0.1, 0.5], dtype=torch.float32)
            self.rb.vel = torch.tensor([0.0, -2.0, 0.0], dtype=torch.float32)
            self.rb.ang_vel = torch.tensor([
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
            ], dtype=torch.float32)
            self.rb.sleep_counter = 0

    def override_physics_step(self, dt_real):
        if not self.pause_physics:
            self.sim_time += dt_real
            steps = 0
            while self.sim_time >= self.dt_phys and steps < 10:
                self.rb.step(self.dt_phys, self.gravity, self.restitution)
                self.sim_time -= self.dt_phys
                steps += 1
            if steps == 10:
                self.sim_time = 0.0
                
        # Fake MPM update so Editor.run() can seamlessly skin and render!
        R_torch = quat_to_matrix(self.rb.quat)
        pos_ti = ti.math.vec3([self.rb.pos[0].item(), self.rb.pos[1].item(), self.rb.pos[2].item()])
        R_ti = ti.math.mat3([
            [R_torch[0,0].item(), R_torch[0,1].item(), R_torch[0,2].item()],
            [R_torch[1,0].item(), R_torch[1,1].item(), R_torch[1,2].item()],
            [R_torch[2,0].item(), R_torch[2,1].item(), R_torch[2,2].item()]
        ])
        
        update_phys_from_rigid_body(self.local_pos_ti, pos_ti, R_ti, self.N)
        
        return True # Tell Editor.run() that we handled the physics!

def main():
    print("[INFO] Starting PhysGaussian Studio (Rigid Body Mode)...")
    loader = ModelLoader(
        max_render_points=phys.MAX_RENDER_POINTS, 
        max_phys_points=phys.MAX_PHYS_POINTS,
        use_mesh=False
    )
    loader._load_worker("modello.ply")
    init_data = loader.get_loaded_data()
    
    editor = RigidBodyEditor(loader, init_data)
    
    # We must call init_data so that phys fields are properly initialized
    phys.init_data(init_data["pos_local_phys"])
    
    editor.run()

if __name__ == "__main__":
    main()
