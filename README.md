# 3D Gaussian Splatting Interactive Demos

This repository contains a collection of interactive scripts and physics simulations built on top of 3D Gaussian Splatting (3DGS). By combining PyTorch-based rasterization with Taichi-based physics engines (like Rigid Body and MPM solvers), these demos allow real-time deformation, fracture, cutting, and relighting of photorealistic 3DGS scans.

## Demos & Scripts

### Standard PLY Viewer (`ply_viewer.py`)
The baseline photorealistic viewer. Use this to examine your `.ply` scans in pristine, high-resolution quality with a free-look camera before processing them through the physics engines.
![PLY Viewer Demo](results/gif/ply_viewer.gif)

### Material Point Method (Soft Body) Simulator (`physgaussian_mpm_cuda_gui_materials.py`)
Utilizes a CUDA-accelerated MPM (Material Point Method) solver to simulate complex elastoplastic materials. Turn your 3D scan into jelly, snow, or soft rubber that squishes, bounces, and tears dynamically.
![MPM Soft Body Demo](results/gif/mpm_shoe.gif)
![MPM Soft Body Demo](results/gif/mpm_pillows.gif)
![MPM Soft Body Demo](results/gif/mpm_ficus.gif)

### Dynamic Relighting (`dynamic_relighting_02_editor.py`)
An editor focused on visual fidelity and scene lighting. This script allows you to manipulate point lights and ambient lighting, simulating real-time specular highlights and shadows on the Gaussian splats.
![Dynamic Relighting Demo](results/gif/relighting.gif)

### Rigid Body Physics Simulator (`rigid_body_3dgs.py`)
This script turns static 3DGS scenes into fully interactive rigid bodies. It automatically generates a collision proxy mesh (convex hull) and drops the object into a real-time physics environment, reacting to gravity, collisions, and the floor.
![Rigid Body Demo](results/gif/rigid_body.gif)

### Laser Cutter (`physgaussian_laser_cutter.py`)
An interactive, mouse-driven laser cutting tool. By clicking and dragging the middle mouse button across the object, you cast a ray into the scene that physically disables and cuts away Gaussians, allowing you to slice the 3D scan in real-time.
![Laser Cutter Demo](results/gif/laser_cutter.gif)

### Effects (`gaussian_effects_editor.py`)
A sandbox environment for manipulating the raw properties of the Gaussian splats (scales, rotations, spherical harmonics) to achieve various visual effects and distortions natively on the scan.
![Gaussian Effects Demo](results/gif/effects.gif)

### Mesh and Voxel Converters (`ply_to_mesh.py`, `ply_to_vox.py`)
Utility scripts used by the physics engines to downsample the massive 3DGS point clouds into manageable proxy meshes (via Alpha Shapes/Convex Hulls) or Voxel grids for efficient collision detection.

![Mesh Converter](results/images/mesh.png)
![Mesh Converter](results/images/vox.png)


## Setup Guide

Follow these steps to set up your Python environment and start running the interactive scripts:

### 1. Create a Virtual Environment
It is highly recommended to use a Python virtual environment to manage dependencies locally. From the root directory of this project, run:
```bash
python -m venv venv
```

### 2. Activate the Environment
You must activate the environment every time before installing packages or running the simulations.

- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
With your virtual environment active, install all required packages (including PyTorch, Taichi, and `diff-gaussian-rasterization`) using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Running the Scripts
Once everything is installed, you can execute any individual demo script directly via Python. Ensure your environment is active, then run:
```bash
# Example: Run the rigid body physics demo
python rigid_body_3dgs.py
```
