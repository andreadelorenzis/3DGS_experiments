import numpy as np
from plyfile import PlyData

def load_3dgs_ply(ply_path, max_points=None):
    """
    Unified PLY loader for 3D Gaussian Splatting data.
    Reads a .ply file, optionally subsamples to `max_points`, and extracts raw numpy arrays for:
    - positions (pos)
    - spherical harmonics (shs)
    - base colors (cols)
    - opacities (opac)
    - scales (scale)
    - rotations (rot)

    Applying sigmoid to opacity and exp to scales, and normalizing quaternions is done here.
    No domain-specific offsets or bounding box scaling are applied, returning the raw arrays.
    Returns a dictionary of the arrays.
    """
    print(f"Loading 3DGS PLY: {ply_path}" + (f" (max {max_points} points)" if max_points else ""))
    try:
        plydata = PlyData.read(ply_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    v = plydata["vertex"]

    if max_points is not None:
        stride = max(1, len(v["x"]) // max_points)
    else:
        stride = 1

    x, y, z = v["x"][::stride], v["y"][::stride], v["z"][::stride]
    pos = np.vstack((x, y, z)).T
    
    fields = v.data.dtype.names

    # Extract Spherical Harmonics
    shs = np.zeros((len(pos), 16, 3), dtype=np.float32)
    SH_C0 = 0.28209479177387814

    if "f_dc_0" in fields:
        shs[:, 0, 0] = v["f_dc_0"][::stride]
        shs[:, 0, 1] = v["f_dc_1"][::stride]
        shs[:, 0, 2] = v["f_dc_2"][::stride]

        r = shs[:, 0, 0] * SH_C0 + 0.5
        g = shs[:, 0, 1] * SH_C0 + 0.5
        b = shs[:, 0, 2] * SH_C0 + 0.5

        if "f_rest_0" in fields:
            for i in range(15):
                shs[:, i + 1, 0] = v[f"f_rest_{i}"][::stride]
                shs[:, i + 1, 1] = v[f"f_rest_{i+15}"][::stride]
                shs[:, i + 1, 2] = v[f"f_rest_{i+30}"][::stride]
    elif "red" in fields and "green" in fields and "blue" in fields:
        r = v["red"][::stride] / 255.0
        g = v["green"][::stride] / 255.0
        b = v["blue"][::stride] / 255.0
        # Dummy SH for base colors
        shs[:, 0, 0] = (r - 0.5) / SH_C0
        shs[:, 0, 1] = (g - 0.5) / SH_C0
        shs[:, 0, 2] = (b - 0.5) / SH_C0
    elif "r" in fields and "g" in fields and "b" in fields:
        r = v["r"][::stride] / 255.0
        g = v["g"][::stride] / 255.0
        b = v["b"][::stride] / 255.0
        shs[:, 0, 0] = (r - 0.5) / SH_C0
        shs[:, 0, 1] = (g - 0.5) / SH_C0
        shs[:, 0, 2] = (b - 0.5) / SH_C0
    else:
        r = g = b = np.full_like(x, 0.5)
        shs[:, 0, 0] = 0.0
        shs[:, 0, 1] = 0.0
        shs[:, 0, 2] = 0.0

    cols = np.clip(np.vstack((r, g, b)).T, 0, 1)

    # Opacity, Scale, Rotation
    try:
        scale = np.vstack((v["scale_0"][::stride], v["scale_1"][::stride], v["scale_2"][::stride])).T
        scale = np.exp(scale)

        rot = np.vstack((v["rot_0"][::stride], v["rot_1"][::stride], v["rot_2"][::stride], v["rot_3"][::stride])).T
        rot = rot / np.linalg.norm(rot, axis=1, keepdims=True)
        
        opac = 1.0 / (1.0 + np.exp(-v["opacity"][::stride]))
    except ValueError:
        scale = np.full_like(pos, 0.01)
        rot = np.zeros((len(pos), 4))
        rot[:, 0] = 1.0
        opac = np.full_like(x, 1.0)
    
    print(f"[OK] 3DGS parameters loaded. Total points extracted: {len(pos)}")

    return {
        "pos": pos.astype(np.float32),
        "shs": shs.astype(np.float32),
        "cols": cols.astype(np.float32),
        "scale": scale.astype(np.float32),
        "rot": rot.astype(np.float32),
        "opac": opac.astype(np.float32),
        "raw_opacity": v["opacity"][::stride] if "opacity" in fields else np.zeros_like(x), # sometimes needed un-sigmoid
        "raw_scale": np.vstack((v["scale_0"][::stride], v["scale_1"][::stride], v["scale_2"][::stride])).T if "scale_0" in fields else np.zeros_like(pos),
        "raw_rot": np.vstack((v["rot_0"][::stride], v["rot_1"][::stride], v["rot_2"][::stride], v["rot_3"][::stride])).T if "rot_0" in fields else np.zeros((len(pos), 4))
    }
