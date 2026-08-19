import torch
import math

def get_world_to_view_matrix(eye, target, device):
    """Generates a World-to-View matrix (LookAt) for PyTorch."""
    forward = target - eye
    forward = forward / torch.norm(forward)
    up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=device)
    right = torch.cross(forward, up, dim=0)
    right = right / torch.norm(right)
    true_up = torch.cross(right, forward, dim=0)

    view = torch.eye(4, dtype=torch.float32, device=device)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = forward
    view[:3, 3] = -torch.matmul(view[:3, :3], eye)
    return view.transpose(0, 1)

def getProjectionMatrix(znear, zfar, fovX, fovY, device):
    """Generates a Perspective Projection Matrix for PyTorch."""
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P.transpose(0, 1)

import numpy as np
import taichi as ti

class FreeCamera:
    def __init__(self, position, target, fovY_deg=60.0):
        self.pos = np.array(position, dtype=np.float32)
        
        # Calculate initial azimuth and elevation from target
        forward = np.array(target, dtype=np.float32) - self.pos
        forward_xz = np.linalg.norm([forward[0], forward[2]])
        
        self.azimuth = math.atan2(forward[0], forward[2])
        self.elevation = math.atan2(forward[1], forward_xz)
        
        self.fovY = fovY_deg * math.pi / 180.0
        self.last_mouse_pos = None

    def update(self, window, speed, dt_real=None):
        """Updates camera position and orientation based on input."""
        # Calculate directional vectors
        forward = np.array(
            [
                math.cos(self.elevation) * math.sin(self.azimuth),
                math.sin(self.elevation),
                math.cos(self.elevation) * math.cos(self.azimuth),
            ],
            dtype=np.float32,
        )
        forward_dir = forward / (np.linalg.norm(forward) + 1e-6)
        
        # Right vector (assuming Y is up/down, -Y is up for 3dgs?)
        # Yes, up = [0, -1, 0]. 
        right = np.cross(forward_dir, [0, -1, 0])
        right /= (np.linalg.norm(right) + 1e-6)

        # Keyboard movement (WASD + Arrows)
        if window.is_pressed('w'):
            self.pos += forward_dir * speed
        if window.is_pressed('s'):
            self.pos -= forward_dir * speed
        if window.is_pressed('d'):
            self.pos += right * speed
        if window.is_pressed('a'):
            self.pos -= right * speed
        
        # Up/down arrows for vertical translation
        if window.is_pressed(ti.ui.UP):
            self.pos[1] -= speed * 2 # Since Y is down, - is visually up
        if window.is_pressed(ti.ui.DOWN):
            self.pos[1] += speed * 2

        # Mouse look (RMB to pan)
        curr_mouse = window.get_cursor_pos()
        if window.is_pressed(ti.ui.RMB):
            if self.last_mouse_pos is not None:
                dx = curr_mouse[0] - self.last_mouse_pos[0]
                dy = curr_mouse[1] - self.last_mouse_pos[1]
                self.azimuth -= dx * 3.0
                self.elevation = max(-1.4, min(1.4, self.elevation + dy * 3.0))
            self.last_mouse_pos = curr_mouse
        else:
            self.last_mouse_pos = None
            
        target = self.pos + forward_dir
        return self.pos, target

    def get_matrices(self, device, W, H, znear=0.01, zfar=100.0):
        """Returns the View and Projection matrices."""
        tan_fovy = math.tan(self.fovY / 2)
        tan_fovx = tan_fovy * (W / H)
        fovX = 2.0 * math.atan(tan_fovx)
        
        forward = np.array(
            [
                math.cos(self.elevation) * math.sin(self.azimuth),
                math.sin(self.elevation),
                math.cos(self.elevation) * math.cos(self.azimuth),
            ],
            dtype=np.float32,
        )
        forward_dir = forward / (np.linalg.norm(forward) + 1e-6)
        target_np = self.pos + forward_dir
        
        t_pos = torch.tensor(self.pos, dtype=torch.float32, device=device)
        t_target = torch.tensor(target_np, dtype=torch.float32, device=device)
        
        viewmatrix = get_world_to_view_matrix(t_pos, t_target, device)
        projmatrix = getProjectionMatrix(znear, zfar, fovX, self.fovY, device)
        
        return viewmatrix, projmatrix, t_pos, t_target, tan_fovx, tan_fovy
