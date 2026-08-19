import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R


def extract_analytical_mesh_colored(
    ply_path, output_mesh_path, opacity_threshold=0.1, poisson_depth=9
):
    print("1. Lettura del file 3DGS .ply...")
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]
    fields = v.data.dtype.names

    # 1. Estrazione Coordinate
    x, y, z = v["x"], v["y"], v["z"]

    # 2. Gestione Opacità (Abbassata per non bucare i dettagli lucidi)
    if "opacity" in fields:
        op = v["opacity"]
        if op.max() > 1.0 or op.min() < 0.0:
            op = 1.0 / (1.0 + np.exp(-op))
    else:
        op = np.ones_like(x)

    mask = op > opacity_threshold

    # 3. Estrazione Colori (Il Fallback)
    if "f_dc_0" in fields:
        SH_C0 = 0.28209
        r = v["f_dc_0"] * SH_C0 + 0.5
        g = v["f_dc_1"] * SH_C0 + 0.5
        b = v["f_dc_2"] * SH_C0 + 0.5
    elif "red" in fields and "green" in fields and "blue" in fields:
        r = v["red"] / 255.0
        g = v["green"] / 255.0
        b = v["blue"] / 255.0
    elif "r" in fields and "g" in fields and "b" in fields:
        r = v["r"] / 255.0
        g = v["g"] / 255.0
        b = v["b"] / 255.0
    else:
        r = g = b = np.full_like(x, 0.5)

    colors = np.vstack((r[mask], g[mask], b[mask])).T
    colors = np.clip(colors, 0, 1)

    # 4. Estrazione Scala e Quaternioni per le Normali
    sx = np.exp(v["scale_0"][mask])
    sy = np.exp(v["scale_1"][mask])
    sz = np.exp(v["scale_2"][mask])

    qw = v["rot_0"][mask]
    qx = v["rot_1"][mask]
    qy = v["rot_2"][mask]
    qz = v["rot_3"][mask]

    points = np.vstack((x[mask], y[mask], z[mask])).T
    print(f"   Punti validi analizzati: {len(points)}")

    print("2. Calcolo Matematico delle Normali...")
    quats = np.vstack((qx, qy, qz, qw)).T
    rot_matrices = R.from_quat(quats).as_matrix()
    normals = np.zeros_like(points)
    scales = np.vstack((sx, sy, sz)).T

    for i in range(len(points)):
        min_axis_idx = np.argmin(scales[i])
        normals[i] = rot_matrices[i, :, min_axis_idx]

    print("3. Generazione nuvola e pulizia delicata...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)  # Assegniamo i colori alla nuvola

    # Reso meno aggressivo (std_ratio alzato a 3.0) per non cancellare i dettagli della scarpa
    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=3.0)
    pcd_clean.orient_normals_consistent_tangent_plane(15)

    print(f"4. Poisson Surface Reconstruction (Depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_clean, depth=poisson_depth, linear_fit=True
    )

    print("5. Taglio artefatti e proiezione dei COLORI...")
    densities = np.asarray(densities)
    # Taglio leggermente più basso per non allargare i buchi
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # --- IL TRUCCO PER COLORARE LA MESH (KD-Tree Transfer) ---
    print("   -> Trasferimento colori dalla nuvola alla mesh...")
    kdtree = o3d.geometry.KDTreeFlann(pcd_clean)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_colors = np.zeros_like(mesh_vertices)

    pcd_clean_colors = np.asarray(pcd_clean.colors)

    for i in range(len(mesh_vertices)):
        # Per ogni vertice della mesh, trova il punto più vicino nella nuvola originale
        _, idx, _ = kdtree.search_knn_vector_3d(mesh_vertices[i], 1)
        # Assegna a quel vertice il colore del punto originale
        mesh_colors[i] = pcd_clean_colors[idx[0]]

    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    # --------------------------------------------------------

    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    print(f"6. Salvataggio Mesh Colorata in: {output_mesh_path}")
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)

    print("Apertura visualizzatore 3D...")
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name="Mesh Colorata 3DGS")


if __name__ == "__main__":
    extract_analytical_mesh_colored(
        ply_path="modello.ply",
        output_mesh_path="mesh_colorata.obj",
        opacity_threshold=0.1,  # Più basso per catturare le strisce
        poisson_depth=9,
    )
