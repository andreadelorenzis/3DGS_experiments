import os
import threading
import numpy as np
from scipy.spatial import cKDTree
from utils.ply import load_3dgs_ply

class ModelLoader:
    def __init__(self, models_dir="models", max_render_points=3000000, max_phys_points=20000, use_mesh=False):
        self.use_mesh = use_mesh
        if self.use_mesh:
            try:
                import open3d as o3d
            except ImportError:
                raise ImportError("open3d is required when use_mesh=True. Install with: pip install open3d")

        self.models_dir = models_dir
        self.max_render_points = max_render_points
        self.max_phys_points = max_phys_points
        self.is_loading = False
        self.loaded_data = None

    def create_lowpoly_mesh(self, points, max_faces=3000):
        import open3d as o3d
        print("[ModelLoader] Creating low-poly mesh (Alpha Shape + Decimation)...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        alpha = 0.08
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        if len(mesh.triangles) > max_faces:
            mesh = mesh.simplify_quadric_decimation(max_faces)

        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.triangles, dtype=np.int32)

        if len(v) < 10:
            print("[ModelLoader] [WARN] Alpha shape not optimal, using Convex Hull.")
            mesh, _ = pcd.compute_convex_hull()
            if len(mesh.triangles) > max_faces:
                mesh = mesh.simplify_quadric_decimation(max_faces)
            v = np.asarray(mesh.vertices, dtype=np.float32)
            f = np.asarray(mesh.triangles, dtype=np.int32)

        print(f"[ModelLoader] [OK] Proxy mesh created: {len(v)} vertices, {len(f)} faces.")
        return v, f


    def list_models(self):
        """Returns a list of .ply files in the models directory."""
        if not os.path.exists(self.models_dir):
            return []
        return sorted([f for f in os.listdir(self.models_dir) if f.endswith(".ply")])

    def async_load_model(self, model_name):
        """Starts a background thread to load a model and process its skinning."""
        if self.is_loading:
            return
        self.is_loading = True
        self.loaded_data = None
        thread = threading.Thread(target=self._load_worker, args=(model_name,))
        thread.start()

    def _load_worker(self, model_name):
        try:
            ply_path = os.path.join(self.models_dir, model_name)
            if not os.path.exists(ply_path):
                # Fallback to current directory if not found in models/
                ply_path = model_name

            print(f"[ModelLoader] Background loading {ply_path}...")
            
            # Load point cloud data from the specified PLY file
            data = load_3dgs_ply(ply_path, max_points=self.max_render_points)
            pos = data["pos"]      # (M, 3) 3D positions of the Gaussian points
            cols = data["cols"]    # (M, 3) Base spherical harmonic colors
            scale = data["scale"]  # (M, 3) 3D scales of the Gaussians
            rot = data["rot"]      # (M, 4) Rotation quaternions for each Gaussian
            opac = data["opac"]    # (M, 1) Opacity value of each Gaussian
            shs = data["shs"]      # (M, K) Higher-order spherical harmonics data

            # Compute bounding box and extent for normalization
            min_bound = pos.min(axis=0) # shape: (3,)
            max_bound = pos.max(axis=0) # shape: (3,)
            extent = (max_bound - min_bound).max() # Scalar

            # Normalize positions and scales so the object fits within a unit cube
            scale = scale / extent
            pos = (pos - min_bound) / extent
            
            # Center the positions locally
            pos_center = pos.mean(axis=0)
            pos_local = pos - pos_center
            
            # Define a base offset for rendering the model
            base_offset = np.array([0.5, 0.64, 0.5])

            # Convert all point attributes to float32 for rendering performance
            render_pos_local = pos_local.astype(np.float32)    # (M, 3) Local render positions
            render_cols_np = cols.astype(np.float32)           # (M, 3) Render colors
            render_scale_np = scale.astype(np.float32)         # (M, 3) Render scales
            render_rot_np = rot.astype(np.float32)             # (M, 4) Render rotations
            render_opac_np = opac.astype(np.float32)           # (M, 1) Render opacities
            render_shs_np = shs.astype(np.float32)             # (M, K) Render SH coefficients

            print("[ModelLoader] Subsampling for physics...")
            mesh_f_np = None
            
            # Specific filtering for pillow2sofa model to prevent invisible obstacles
            is_pillow = "pillow2sofa" in ply_path.lower()
            min_opac = 0.5 if is_pillow else 0.05
            
            valid_indices = np.where(opac > min_opac)[0]
            if len(valid_indices) > 0:
                render_pos_local = render_pos_local[valid_indices]
                render_cols_np = render_cols_np[valid_indices]
                render_scale_np = render_scale_np[valid_indices]
                render_rot_np = render_rot_np[valid_indices]
                render_opac_np = render_opac_np[valid_indices]
                render_shs_np = render_shs_np[valid_indices]
            
            if self.use_mesh:
                # 1. Generate Low-Poly Proxy Mesh
                proxy_v_np, proxy_f_np = self.create_lowpoly_mesh(render_pos_local, max_faces=2500)
                mesh_f_np = proxy_f_np

                # 2. Generate INTERNAL physical points (Volumetric Support)
                is_hollow = "pillow2sofa" in ply_path.lower()
                if True: # if not is_hollow:
                    print("[ModelLoader] Generating internal support points...")
                    phys_stride = max(1, len(render_pos_local) // (self.max_phys_points - len(proxy_v_np)))
                    internal_np = render_pos_local[::phys_stride].copy()
                    # Combine mesh vertices and internal points into a single array for MPM PHYSICS
                    pos_local_phys = np.vstack((proxy_v_np, internal_np)).astype(np.float32)
                else:
                    print(f"[ModelLoader] Skipping internal support points for {ply_path}...")
                    pos_local_phys = proxy_v_np.astype(np.float32)
                
                if len(pos_local_phys) > self.max_phys_points:
                    pos_local_phys = pos_local_phys[:self.max_phys_points]

                print("[ModelLoader] Computing skinning (KDTree exclusively on surface mesh vertices)...")
                _tree = cKDTree(proxy_v_np)
                _, skin_idx_np = _tree.query(render_pos_local, k=1)
            else:
                # To prevent MPM density instability (which causes the ficus to break),
                # we MUST sample the points uniformly across the surface. 
                # Stacking two different samplings (base + internal) created mass clumps on the surface.
                target_points = self.max_phys_points
                
                if len(render_pos_local) > target_points:
                    phys_indices = np.linspace(0, len(render_pos_local) - 1, target_points).astype(int)
                else:
                    phys_indices = np.arange(len(render_pos_local))
                pos_local_phys = render_pos_local[phys_indices].copy()
                
                print(f"[ModelLoader] Final physical particles: {len(pos_local_phys)}")
                print("[ModelLoader] Computing skinning (KDTree on all physical particles)...")
                _tree = cKDTree(pos_local_phys)
                _, skin_idx_np = _tree.query(render_pos_local, k=1)

            skin_idx_np = skin_idx_np.astype(np.int64)

            # Compute the local offset of each render point relative to its driving physics particle
            local_offset_canonical_np = (render_pos_local - pos_local_phys[skin_idx_np]).astype(np.float32)

            # Store the final model components including rendering attributes, physics particles, and skinning data
            self.loaded_data = {
                "render_pos_local": render_pos_local,                   # (M, 3) Local render point positions
                "render_cols_np": render_cols_np,                       # (M, 3) Render base colors
                "render_scale_np": render_scale_np,                     # (M, 3) Render Gaussian scales
                "render_rot_np": render_rot_np,                         # (M, 4) Render Gaussian rotations
                "render_opac_np": render_opac_np,                       # (M, 1) Render Gaussian opacities
                "render_shs_np": render_shs_np,                         # (M, K) Render Gaussian SHs
                "base_offset": base_offset,                             # (3,) Global translation offset for rendering
                "pos_local_phys": pos_local_phys,                       # (N, 3) Subsampled physics particle positions
                "skin_idx_np": skin_idx_np,                             # (M,) Physics particle index for each render point
                "local_offset_canonical_np": local_offset_canonical_np, # (M, 3) Canonical offset from physics particle to render point
                "N": len(pos_local_phys),                               # (int) Total number of physics particles
                "M": len(render_pos_local),                             # (int) Total number of render points
                "mesh_f_np": mesh_f_np,                                 # Mesh faces for rendering (if use_mesh=True)
                "extent": extent                                        # (float) Original spatial extent
            }
            print("[ModelLoader] Model loaded successfully.")
        except Exception as e:
            print(f"[ModelLoader] Error loading model: {e}")
        finally:
            self.is_loading = False

    def get_loaded_data(self):
        """Returns the loaded data and clears it, or None if not ready."""
        data = self.loaded_data
        self.loaded_data = None
        return data
