# view_ply.py — auto-fits the whole scene so both people are visible
import os
import numpy as np
import open3d as o3d

# ========= EDIT THESE =========
PLY_PATH    = r"C:\Users\sybanzon\Documents\schoolFiles\RESERARCH\full\results\output_point_cloud.ply"
CALIB_NPZ   = r"C:\Users\sybanzon\Documents\schoolFiles\RESERARCH\calibration\v0.npz"  # leave "" to skip cam alignment
CLOUD_FRAME = "original"   # "original" or "rectified" -> frame your PLY points are in
VIEW_AS     = "original"   # "original" or "rectified" -> camera you want to view from
MATCH_WINDOW_TO_CALIB = True
WINDOW_W, WINDOW_H = 1280, 720
AXES_SIZE   = 0.1
# Fit controls
MARGIN_SCALE   = 1.10   # 10% extra space around the whole cloud
NEAR_MARGIN_Z  = 0.05   # keep nearest point at least this far from the camera (units of your cloud)
REMOVE_OUTLIERS = True  # do a quick outlier removal to avoid a few stray points ruining the fit
# ==============================

def load_calib(npz_path: str):
    data = np.load(npz_path)
    K   = data["camera_matrix_1"]        # 3x3
    R1  = data["R1"]                     # 3x3
    imw, imh = map(int, data["image_size"])
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    return imw, imh, fx, fy, cx, cy, R1

def build_extrinsic(R_w2c: np.ndarray, C_world: np.ndarray) -> np.ndarray:
    # world->camera: Xc = R (Xw - C) = R Xw - R C, so t = -R C
    E = np.eye(4, dtype=np.float64)
    E[:3,:3] = R_w2c
    E[:3, 3] = -R_w2c @ C_world
    return E

def compute_fit_camera_extrinsic(pcd: o3d.geometry.PointCloud,
                                 R_w2c: np.ndarray,
                                 fx: float, fy: float,
                                 width: int, height: int) -> np.ndarray:
    """
    Choose a camera position so the whole cloud fits the image given intrinsics.
    Keeps orientation R_w2c. Centers on the scene. Pushes camera back enough that
    both width and height fit, with a margin, and avoids near-plane clipping.
    """
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        return build_extrinsic(R_w2c, np.zeros(3))

    # Optionally remove a few outliers
    if REMOVE_OUTLIERS and pts.shape[0] > 1000:
        pcd_tmp = pcd.voxel_down_sample(voxel_size=max(1e-6, pcd.get_min_bound().__abs__().max()*1e-6))
        pcd_tmp, _ = pcd_tmp.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        pts = np.asarray(pcd_tmp.points, dtype=np.float64)

    # Orientation fixed. We will only choose camera center C.
    # Transform points to camera orientation only (no translation).
    P_cam_oriented = pts @ R_w2c.T

    # Use the AABB center in camera coords for centering
    mins = P_cam_oriented.min(axis=0)
    maxs = P_cam_oriented.max(axis=0)
    center_cam = 0.5 * (mins + maxs)

    # Half extents around the center
    rx = max(abs(maxs[0] - center_cam[0]), abs(center_cam[0] - mins[0]))
    ry = max(abs(maxs[1] - center_cam[1]), abs(center_cam[1] - mins[1]))

    # Field of view from intrinsics
    fov_x = 2.0 * np.arctan2(width, 2.0*fx)
    fov_y = 2.0 * np.arctan2(height, 2.0*fy)

    # Distance so that the half width/height fits the FOV with margin
    d_x = (rx * MARGIN_SCALE) / np.tan(fov_x * 0.5) if fov_x > 1e-6 else 1e6
    d_y = (ry * MARGIN_SCALE) / np.tan(fov_y * 0.5) if fov_y > 1e-6 else 1e6
    d = max(d_x, d_y)

    # Keep nearest point off the near plane
    nearest_z = P_cam_oriented[:,2].min() - center_cam[2] + d
    if nearest_z < NEAR_MARGIN_Z:
        d += (NEAR_MARGIN_Z - nearest_z)

    # Camera forward axis in world coords is z_cam in world = R^T * [0,0,1]
    z_cam_world = R_w2c.T @ np.array([0.0, 0.0, 1.0])

    # World center corresponding to center_cam
    # Since we used orientation only, world center is R^T * center_cam
    center_world = R_w2c.T @ center_cam

    # Place camera so that center maps to image center at depth d
    C_world = center_world - d * z_cam_world

    return build_extrinsic(R_w2c, C_world)

def view_with_params(pcd: o3d.geometry.PointCloud,
                     width: int, height: int,
                     fx: float, fy: float, cx: float, cy: float,
                     extrinsic_4x4: np.ndarray,
                     title: str):
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intr
    params.extrinsic = extrinsic_4x4

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=width, height=height, visible=True)
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXES_SIZE))

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    # Relax near/far if possible and make points a bit bigger
    try:
        ctr.set_constant_z_near(0.01)
        ctr.set_constant_z_far(1e6)
    except Exception:
        pass
    opt = vis.get_render_option()
    opt.point_size = 2.0

    vis.run()
    vis.destroy_window()

def simple_view(pcd: o3d.geometry.PointCloud, title: str):
    o3d.visualization.draw_geometries(
        [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXES_SIZE)],
        window_name=title, width=WINDOW_W, height=WINDOW_H
    )

def main():
    if not os.path.isfile(PLY_PATH):
        print(f"PLY not found:\n  {PLY_PATH}")
        return

    pcd = o3d.io.read_point_cloud(PLY_PATH)
    if pcd.is_empty():
        print("Loaded file, but it has no points.")
        return

    if not CALIB_NPZ or not os.path.isfile(CALIB_NPZ):
        if CALIB_NPZ and not os.path.isfile(CALIB_NPZ):
            print(f"(Warning) Calibration file not found, using simple viewer:\n  {CALIB_NPZ}")
        simple_view(pcd, f"Open3D: {PLY_PATH}")
        return

    # Load intrinsics and rectification rotation
    imw, imh, fx, fy, cx, cy, R1 = load_calib(CALIB_NPZ)
    width  = imw if MATCH_WINDOW_TO_CALIB else WINDOW_W
    height = imh if MATCH_WINDOW_TO_CALIB else WINDOW_H

    # Orientation depends on frame choice
    if VIEW_AS == "original" and CLOUD_FRAME == "rectified":
        R_w2c = R1.T
    elif VIEW_AS == "rectified" and CLOUD_FRAME == "original":
        R_w2c = R1
    else:
        R_w2c = np.eye(3, dtype=np.float64)

    # Compute camera placement that fits the full scene
    E = compute_fit_camera_extrinsic(pcd, R_w2c, fx, fy, width, height)

    view_with_params(
        pcd,
        width=width, height=height,
        fx=fx, fy=fy, cx=cx, cy=cy,
        extrinsic_4x4=E,
        title=f"Open3D: {PLY_PATH}"
    )

if __name__ == "__main__":
    main()
