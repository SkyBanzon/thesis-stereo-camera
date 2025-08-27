#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depth map generation from a 3840x1080 side-by-side stereo image using calibration from opencv_stereo_params.npz.

- Loads calibration from npz (K1, D1, K2, D2, R1, R2, P1, P2, Q, image_size)
- Rectifies 1920x1080 per-eye frames
- Computes disparity via StereoSGBM (WLS optional)
- Converts disparity to metric depth (Z = f * B / d)
- Saves disparity maps and depth map (npy + 16-bit PNG)
- NEW: Saves a metric point cloud (PLY) via PointCloudGenerator (your class)
"""

import os
import numpy as np
import cv2
from typing import Tuple, Optional

# ---------------------------
# Load calibration
# ---------------------------
def load_npz_calibration(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    params = np.load(npz_path, allow_pickle=True)
    calib = {
        "K1": params["camera_matrix_1"],
        "D1": params["dist_coeffs_1"].ravel(),
        "K2": params["camera_matrix_2"],
        "D2": params["dist_coeffs_2"].ravel(),
        "R1": params["R1"],
        "R2": params["R2"],
        "P1": params["P1"],
        "P2": params["P2"],
        "Q":  params["Q"],
        "per_eye_size": tuple(params["image_size"])
    }
    return calib

# ---------------------------
# SBS load & rectification
# ---------------------------
def load_sbs(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    left  = img[:, : w//2].copy()
    right = img[:,  w//2 :].copy()
    return left, right

def rectify(left_bgr, right_bgr, calib: dict):
    W, H = calib["per_eye_size"]
    map1x, map1y = cv2.initUndistortRectifyMap(calib["K1"], calib["D1"], calib["R1"], calib["P1"], (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(calib["K2"], calib["D2"], calib["R2"], calib["P2"], (W, H), cv2.CV_32FC1)
    rectL = cv2.remap(left_bgr,  map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(right_bgr, map2x, map2y, cv2.INTER_LINEAR)
    return rectL, rectR

# ---------------------------
# Disparity
# ---------------------------
def compute_disparity(rectL_bgr, rectR_bgr, P1, P2, zmin_m=0.1, block_size=15, use_wls=True):
    grayL = cv2.cvtColor(rectL_bgr, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR_bgr, cv2.COLOR_BGR2GRAY)

    # Gentle preprocessing
    grayL = cv2.bilateralFilter(grayL, 9, 50, 50)
    grayR = cv2.bilateralFilter(grayR, 9, 50, 50)

    fx = float(P1[0,0])
    baseline_m = -float(P2[0,3]) / fx
    max_disp_needed = max(32.0, (fx * baseline_m) / float(zmin_m))
    num_disp = int(np.ceil(1.10 * max_disp_needed / 16.0) * 16)
    num_disp = max(64, min(num_disp, 320))  # clamp
    min_disp = 0

    cn = 1
    P1sgbm = 8  * cn * (block_size ** 2)
    P2sgbm = 32 * cn * (block_size ** 2)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1sgbm, P2=P2sgbm,
        disp12MaxDiff=1,
        uniquenessRatio=20,
        speckleWindowSize=200,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    dispL16 = sgbm.compute(grayL, grayR)

    if use_wls and hasattr(cv2, "ximgproc"):
        right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
        dispR16 = right_matcher.compute(grayR, grayL)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
        wls.setLambda(15000.0)
        wls.setSigmaColor(1.2)
        disp = wls.filter(dispL16, rectL_bgr, disparity_map_right=dispR16)
        disp = disp.astype(np.float32) / 16.0
    else:
        disp = dispL16.astype(np.float32) / 16.0

    d16 = (disp * 16.0).astype(np.int16)
    cv2.filterSpeckles(d16, 0, 100, 2 * 16)
    disp = d16.astype(np.float32) / 16.0
    disp[disp < 0] = 0.0
    return disp, fx, baseline_m

# ---------------------------
# Depth from disparity
# ---------------------------
def disparity_to_depth(disp: np.ndarray, fx: float, baseline_m: float):
    Z = np.zeros_like(disp, dtype=np.float32)
    mask = disp > 0
    Z[mask] = (fx * baseline_m) / disp[mask]
    return Z

# ---------------------------
# Visualization
# ---------------------------
def normalize_disp_for_display(disp, p_lo=0.5, p_hi=99.0):
    mask = disp > 0
    vis = np.zeros_like(disp, dtype=np.uint8)
    if np.any(mask):
        v = disp[mask]
        lo = np.percentile(v, p_lo)
        hi = np.percentile(v, p_hi)
        hi = max(hi, lo+1e-3)
        v = np.clip(v, lo, hi)
        vis[mask] = np.round(255.0 * (v - lo) / (hi - lo)).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return vis, color

# ---------------------------
# Point Cloud (YOUR CLASS)
# ---------------------------
class PointCloudGenerator:
    """Handles 3D point cloud generation from disparity maps."""
    
    def __init__(self, Q: np.ndarray):
        """
        Initialize with rectification matrix Q.
        
        Args:
            Q: 4x4 rectification matrix from stereo calibration
        """
        self.Q = Q
    
    def generate_point_cloud(self, disparity: np.ndarray, color_image: np.ndarray, 
                           min_disparity_threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D point cloud from disparity map.
        
        Args:
            disparity: Disparity map
            color_image: Corresponding color image
            min_disparity_threshold: Minimum disparity threshold (if None, uses disparity.min())
            
        Returns:
            Tuple of (3D_points, colors)
        """
        # Reproject points to 3D space
        points_3D = cv2.reprojectImageTo3D(disparity, self.Q)
        
        # Create mask for valid disparities
        if min_disparity_threshold is None:
            min_disparity_threshold = disparity.min()
        
        mask = disparity > min_disparity_threshold
        
        # Apply mask to points and colors
        output_points = points_3D[mask]
        output_colors = color_image[mask]
        
        return output_points, output_colors
    
    @staticmethod
    def save_point_cloud_ply(filename: str, points: np.ndarray, colors: np.ndarray):
        """
        Save point cloud to PLY file format.
        
        Args:
            filename: Output filename
            points: 3D points array
            colors: Color array (BGR format)
        """
        points = points.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        
        ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
end_header
'''
        
        verts = np.hstack([points, colors])
        
        with open(filename, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d')

# ---------------------------
# Main runner
# ---------------------------
def run(image_path, npz_path, out_dir, zmin_m=0.1, use_wls=True):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    calib = load_npz_calibration(npz_path)
    left_bgr, right_bgr = load_sbs(image_path)
    rectL, rectR = rectify(left_bgr, right_bgr, calib)

    disp, fx, baseline_m = compute_disparity(rectL, rectR, calib["P1"], calib["P2"], zmin_m=zmin_m, block_size=15, use_wls=use_wls)
    depth_m = disparity_to_depth(disp, fx, baseline_m)
    disp_u8, disp_col = normalize_disp_for_display(disp)
    
    # Save usual outputs
    cv2.imwrite(os.path.join(out_dir, f"{base}_rect_left.png"), rectL)
    cv2.imwrite(os.path.join(out_dir, f"{base}_rect_right.png"), rectR)
    cv2.imwrite(os.path.join(out_dir, f"{base}_disparity_gray.png"), disp_u8)
    cv2.imwrite(os.path.join(out_dir, f"{base}_disparity_color.png"), disp_col)

    np.save(os.path.join(out_dir, f"{base}_depth_meters.npy"), depth_m)
    depth_mm = np.clip(depth_m*1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(out_dir, f"{base}_depth_mm16.png"), depth_mm)

    # ---- Point cloud (metric) ----
    pcg = PointCloudGenerator(calib["Q"])
    # Using a small threshold (>0) to ignore zeros; adjust if you like
    points, colors = pcg.generate_point_cloud(disp, rectL, min_disparity_threshold=0.0)
    ply_path = os.path.join(out_dir, "output_point_cloud.ply")  # matches your viewer’s default filename
    pcg.save_point_cloud_ply(ply_path, points, colors)

    print(f"[OK] Saved outputs in {out_dir}")
    if np.any(disp > 0):
        print(f" fx={fx:.2f}px, baseline={baseline_m*1000:.2f} mm, disparity range={disp[disp>0].min():.2f}..{disp.max():.2f}")
    print(f" Point cloud (metric): {ply_path}")

if __name__ == "__main__":
    # ==== EDIT HERE ====
    IMAGE_PATH = r"C:\Users\sybanzon\Documents\schoolFiles\RESERARCH\calibration\WIN_20250708_14_23_01_Pro.jpg"   # 3840x1080 SBS
    CALIB_NPZ  = r"C:\Users\sybanzon\Documents\schoolFiles\RESERARCH\calibration\v0.npz"
    OUTPUT_DIR = r"C:\Users\sybanzon\Documents\schoolFiles\RESERARCH\full\results"
    # ===================

    run(IMAGE_PATH, CALIB_NPZ, OUTPUT_DIR, zmin_m=0.1, use_wls=True)
