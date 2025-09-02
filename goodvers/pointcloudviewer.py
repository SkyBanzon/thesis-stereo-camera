# view_pointcloud.py
import numpy as np
import open3d as o3d

# ======================
# EDIT THIS PATH
PLY_FILE = r"C:\Users\sybanzon\Documents\schoolFiles\RESERARCH\full\sept1\output_point_cloud.ply"
# PLY_FILE = "data/output/depth_only/WIN_20250708_14_23_01_Pro_cloud_pyramid.ply"
# ======================

def color_by_height(pcd, axis="z"):
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    h = pts[:, idx]
    h_min, h_max = float(h.min()), float(h.max())
    if h_max <= h_min:
        h_max = h_min + 1e-6
    t = (h - h_min) / (h_max - h_min)
    r = np.clip(1.5 - np.abs(4.0 * (t - 0.75)), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * (t - 0.50)), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * (t - 0.25)), 0, 1)
    cols = np.stack([r, g, b], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(cols)

def main():
    pcd = o3d.io.read_point_cloud(PLY_FILE)
    if pcd.is_empty():
        raise SystemExit(f"Failed to load or empty: {PLY_FILE}")

    # Example: downsample (set to 0 to disable)
    voxel_size = 0.0
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # Example: color by height (set to False to use PLY colors)
    use_height_colors = False
    if use_height_colors:
        color_by_height(pcd, axis="z")

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Viewer - {PLY_FILE}", width=1280, height=720)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0, 0, 0])  # dark background
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
