import torch
import numpy as np
import taichi as ti
import os
from plyfile import PlyData
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

ti.init(arch=ti.cuda)


from utils.ply import load_3dgs_ply
from utils.camera import get_world_to_view_matrix, getProjectionMatrix, FreeCamera
from utils.video_exporter import VideoExporter


# ==========================================
# 3. VIEWER PURO
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Caricamento dati grezzi senza alterazioni di scala o offset
    data = load_3dgs_ply("modello.ply")
    if data is None:
        return
    pos_np, shs_np, opac_np, scale_np, rot_np = data["pos"], data["shs"], data["opac"], data["scale"], data["rot"]

    # 2. Calcolo del bounding box per inquadrare automaticamente il modello
    center_np = pos_np.mean(axis=0)
    extent = float((pos_np.max(axis=0) - pos_np.min(axis=0)).max())

    # 3. Trasferimento su GPU
    t_means3D = torch.tensor(pos_np, device=device)
    t_shs = torch.tensor(shs_np, device=device)
    t_opac = torch.tensor(opac_np, device=device).unsqueeze(1)
    t_scales = torch.tensor(scale_np, device=device)
    t_rots = torch.tensor(rot_np, device=device)

    # 4. Inizializzazione Interfaccia Taichi
    W, H = 1280, 720
    window = ti.ui.Window(
        "Puro 3DGS Viewer (Qualita' Nativa)", res=(W, H), show_window=True
    )
    canvas = window.get_canvas()
    gui = window.get_gui()

    # Adattiamo la distanza della telecamera alla scala REALE del modello
    cam_pos_np = np.array([0.5, 0.4, 1.3], dtype=np.float32)
    target_np = np.array([0.5, 0.4, 0.5], dtype=np.float32)
    camera = FreeCamera(position=cam_pos_np, target=target_np)

    video_exporter = VideoExporter()
    auto_pan = False
    pan_angle = 0.0
    pan_radius = extent * 1.5

    print("\n[INFO] 3D Viewer launched.")
    print(" - LMB (Left Mouse Button) + Drag: Rotate Camera")
    print(" - WASD: Move camera forward/left/back/right")
    print(" - UP/DOWN arrows: Move camera up and down")
    print(" - Q / E: Allontana / Avvicina la telecamera.")

    while window.running:
        # La velocità di movimento scala in base alla grandezza del modello
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
        t_target = torch.tensor(target_np, dtype=torch.float32, device=device)

        viewmatrix, projmatrix, _, _, tan_fovx, tan_fovy = camera.get_matrices(
            device, W, H, znear=0.01, zfar=10.0
        )
        fovY = camera.fovY
        fovX = 2.0 * math.atan(tan_fovx)
        full_proj = viewmatrix @ projmatrix

        # Rendering con Rasterizer Nativo (SH_degree=3 per massima qualità)
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

        # Trasferimento all'interfaccia Taichi
        img_np = image.detach().cpu().numpy()
        img_np = img_np.transpose(2, 1, 0)
        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        canvas.set_image(img_np)

        # Overlay informativo
        gui.begin("Info Visualizzatore", 0.02, 0.02, 0.35, 0.25)
        gui.text(f"Gaussiane: {len(pos_np):,}")
        gui.text("Usa Q/E per Zoom")
        
        if not video_exporter.recording:
            if gui.button("Inizia Registrazione"):
                script_name = os.path.splitext(os.path.basename(__file__))[0]
                import time
                timestamp = time.strftime("%Y%m%d-%H%M%S")
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
