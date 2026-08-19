import struct
import numpy as np
import open3d as o3d
from plyfile import PlyData


def create_vox_with_colors(vox_path, voxel_coords, voxel_colors, grid_size):
    """Genera il file .vox scrivendo anche la palette di colori (Chunk RGBA)."""
    num_voxels = len(voxel_coords)

    # 1. Quantizzazione dei colori (Creiamo una palette 6x6x6 = 216 colori)
    # I colori di Open3D sono da 0.0 a 1.0. Li mappiamo su 6 livelli (0-5)
    quantized = np.round(voxel_colors * 5).astype(int)
    quantized = np.clip(quantized, 0, 5)

    # L'indice del colore andrà da 1 a 216 (0 è vuoto in MagicaVoxel)
    color_indices = quantized[:, 0] * 36 + quantized[:, 1] * 6 + quantized[:, 2] + 1

    # 2. Costruzione della Palette RGBA (256 * 4 bytes)
    palette_data = bytearray(256 * 4)
    for r in range(6):
        for g in range(6):
            for b in range(6):
                idx = r * 36 + g * 6 + b
                palette_data[idx * 4 + 0] = int(r * 255 / 5)  # Rosso
                palette_data[idx * 4 + 1] = int(g * 255 / 5)  # Verde
                palette_data[idx * 4 + 2] = int(b * 255 / 5)  # Blu
                palette_data[idx * 4 + 3] = 255  # Alpha (Opacità)

    # 3. Creazione dei Chunk Binari
    size_content = struct.pack("<iii", grid_size[0], grid_size[1], grid_size[2])
    size_chunk = b"SIZE" + struct.pack("<ii", len(size_content), 0) + size_content

    # Chunk XYZI (Coordinate e colore di ogni voxel)
    xyzi_list = [struct.pack("<i", num_voxels)]
    for i in range(num_voxels):
        x, y, z = voxel_coords[i]
        c = color_indices[i]
        xyzi_list.append(struct.pack("BBBB", int(x), int(y), int(z), int(c)))
    xyzi_content = b"".join(xyzi_list)
    xyzi_chunk = b"XYZI" + struct.pack("<ii", len(xyzi_content), 0) + xyzi_content

    # Chunk RGBA (Inietta la nostra palette)
    rgba_chunk = b"RGBA" + struct.pack("<ii", len(palette_data), 0) + bytes(palette_data)

    main_children = size_chunk + xyzi_chunk + rgba_chunk
    main_chunk = b"MAIN" + struct.pack("<ii", 0, len(main_children)) + main_children

    with open(vox_path, "wb") as f:
        f.write(b"VOX " + struct.pack("<i", 150) + main_chunk)


def convert_to_vox_colored(ply_path, vox_path, resolution=128):
    print("1. Lettura punti e colori dal 3DGS...")
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]

    x, y, z = v["x"], v["y"], v["z"]
    points = np.vstack((x, y, z)).T

    # --- SISTEMA DI FALLBACK PER I COLORI ---
    fields = v.data.dtype.names

    if "f_dc_0" in fields:
        print("   -> Trovate Armoniche Sferiche (f_dc_*). Conversione in RGB...")
        SH_C0 = 0.28209
        r = v["f_dc_0"] * SH_C0 + 0.5
        g = v["f_dc_1"] * SH_C0 + 0.5
        b = v["f_dc_2"] * SH_C0 + 0.5
    elif "red" in fields and "green" in fields and "blue" in fields:
        print("   -> Trovati colori standard (red, green, blue).")
        # I colori PLY standard sono da 0 a 255. Li normalizziamo per Open3D
        r = v["red"] / 255.0
        g = v["green"] / 255.0
        b = v["blue"] / 255.0
    elif "r" in fields and "g" in fields and "b" in fields:
        print("   -> Trovati colori standard (r, g, b).")
        r = v["r"] / 255.0
        g = v["g"] / 255.0
        b = v["b"] / 255.0
    else:
        print("   -> NESSUN COLORE TROVATO. Assegnazione colore grigio di default.")
        r = np.full_like(x, 0.5)
        g = np.full_like(x, 0.5)
        b = np.full_like(x, 0.5)

    colors = np.vstack((r, g, b)).T
    colors = np.clip(colors, 0, 1)  # Assicura che i colori siano tra 0 e 1
    # ----------------------------------------

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("2. Pulizia floaters per massimizzare la risoluzione...")

    # Questo passaggio isola l'oggetto vero ed elimina i detriti dello scan
    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    print(f"3. Voxelizzazione (Griglia massima: {resolution}^3)...")
    # Allinea l'oggetto allo zero (0,0,0)
    min_bound = pcd_clean.get_min_bound()
    pcd_clean.translate(-min_bound)

    # Calcola la dimensione corretta del voxel
    max_extent = max(pcd_clean.get_max_bound())
    voxel_size = max_extent / float(resolution - 1)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd_clean, voxel_size=voxel_size
    )

    voxels = voxel_grid.get_voxels()
    print(f"   Generati {len(voxels)} voxel solidi e colorati!")

    # Estrae coordinate e colori della griglia
    voxel_coords = []
    voxel_colors = []
    max_x = max_y = max_z = 0

    for vx in voxels:
        g_idx = vx.grid_index
        voxel_coords.append(g_idx)
        voxel_colors.append(vx.color)
        if g_idx[0] > max_x:
            max_x = g_idx[0]
        if g_idx[1] > max_y:
            max_y = g_idx[1]
        if g_idx[2] > max_z:
            max_z = g_idx[2]

    voxel_coords = np.array(voxel_coords)
    voxel_colors = np.array(voxel_colors)
    grid_size = (max_x + 1, max_y + 1, max_z + 1)

    print(f"4. Esportazione File MagicaVoxel in: {vox_path}")
    create_vox_with_colors(vox_path, voxel_coords, voxel_colors, grid_size)
    print("PROCESSO COMPLETATO! Aprilo su MagicaVoxel.")


if __name__ == "__main__":
    # Puoi alzare resolution a 256 se il PC lo regge e vuoi più dettagli
    convert_to_vox_colored("modello.ply", "modello_finale_colorato.vox", resolution=128)
