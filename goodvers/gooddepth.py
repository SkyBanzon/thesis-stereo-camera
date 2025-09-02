#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Tuple, Optional
import numpy as np
import cv2

# ========== Calibration Loader ==========
class StereoCalibrationLoader:
    """Loads K/D/R/T from NPZ and computes R1/R2/P1/P2/Q with stereoRectify if missing."""
    def __init__(self, params_file: str):
        self.params_file = params_file
        self.params = None
        self._load_parameters()

    def _get(self, primary, fallback=None):
        if primary in self.params: return self.params[primary]
        if fallback and fallback in self.params: return self.params[fallback]
        raise KeyError(f"Missing key '{primary}'" + (f" (also tried '{fallback}')" if fallback else ""))

    def _load_parameters(self):
        if not os.path.exists(self.params_file):
            raise FileNotFoundError(f"Calibration file not found: {self.params_file}")

        # Load raw
        self.params = np.load(self.params_file, allow_pickle=False)

        # Intrinsics / distortion (handle either naming scheme)
        self.camera_matrix_1 = self._get('camera_matrix_1', 'K1')
        self.camera_matrix_2 = self._get('camera_matrix_2', 'K2')
        self.dist_coeffs_1   = self._get('dist_coeffs_1',  'D1')
        self.dist_coeffs_2   = self._get('dist_coeffs_2',  'D2')

        # Extrinsics (rotation from cam1->cam2, translation in same units)
        self.R  = self._get('R')
        self.T  = self._get('T')
        if self.T.shape == (1, 3):
            self.T = self.T.reshape(3, 1)
        elif self.T.shape == (3,):
            self.T = self.T.reshape(3, 1)

        # Image size (expect (W,H); if (H,W) it’ll still work but we’ll sanity check later)
        self.image_size = tuple(map(int, self._get('image_size').tolist()))
        if len(self.image_size) != 2:
            raise ValueError(f"image_size should be (W,H); got {self.image_size}")

        # If rectification already present use it, else compute it
        if all(k in self.params for k in ['R1', 'R2', 'P1', 'P2', 'Q']):
            self.R1 = self.params['R1']; self.R2 = self.params['R2']
            self.P1 = self.params['P1']; self.P2 = self.params['P2']
            self.Q  = self.params['Q']
        else:
            # Compute rectification (alpha=0 crops, CALIB_ZERO_DISPARITY keeps principal points aligned)
            W, H = self.image_size
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                self.camera_matrix_1, self.dist_coeffs_1,
                self.camera_matrix_2, self.dist_coeffs_2,
                (W, H), self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
            )
            self.R1, self.R2, self.P1, self.P2, self.Q = R1, R2, P1, P2, Q

            # (Optional) quick print to sanity-check baseline sign/scale
            fx = float(self.P1[0,0]) if self.P1[0,0] != 0 else 1.0
            baseline = -float(self.P2[0,3]) / fx  # units follow T
            print(f"[stereoRectify] baseline ≈ {baseline:.4f} (units of T)  size={W}x{H}")

    def generate_rectification_maps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        W, H = self.image_size
        map1x, map1y = cv2.initUndistortRectifyMap(
            self.camera_matrix_1, self.dist_coeffs_1, self.R1, self.P1, (W, H), cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            self.camera_matrix_2, self.dist_coeffs_2, self.R2, self.P2, (W, H), cv2.CV_32FC1
        )
        return map1x, map1y, map2x, map2y

# ========== Image Processor ==========
class StereoImageProcessor:
    def __init__(self, calibration_loader: StereoCalibrationLoader):
        self.calibration = calibration_loader
        self.map1x, self.map1y, self.map2x, self.map2y = self.calibration.generate_rectification_maps()

    def load_combined_stereo_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Combined stereo image not found: {image_path}")
        h, w = img.shape[:2]
        if w % 2 != 0:
            raise ValueError(f"Combined image width must be even; got {w}")
        half = w // 2
        return img[:, :half], img[:, half:]

    def load_separate_images(self, left_path: str, right_path: str) -> Tuple[np.ndarray, np.ndarray]:
        L = cv2.imread(left_path,  cv2.IMREAD_COLOR)
        R = cv2.imread(right_path, cv2.IMREAD_COLOR)
        if L is None: raise FileNotFoundError(f"Left image not found: {left_path}")
        if R is None: raise FileNotFoundError(f"Right image not found: {right_path}")
        return L, R

    def rectify_images(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Ensure images are the same size the calibration expects
        W, H = self.calibration.image_size  # (width, height)
        if (img_left.shape[1], img_left.shape[0]) != (W, H):
            img_left  = cv2.resize(img_left,  (W, H), interpolation=cv2.INTER_LINEAR)
            img_right = cv2.resize(img_right, (W, H), interpolation=cv2.INTER_LINEAR)

        rect_left  = cv2.remap(img_left,  self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return rect_left, rect_right


    def preprocess_for_matching(self, rect_left: np.ndarray, rect_right: np.ndarray,
                                equalize_hist: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        gL = cv2.cvtColor(rect_left,  cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
        if equalize_hist:
            gL = cv2.equalizeHist(gL)
            gR = cv2.equalizeHist(gR)
        return gL, gR

# ========== Disparity / Depth ==========
class DepthMapGenerator:
    """Stereo SGBM + WLS; returns float32 pixel disparity suitable for cv2.reprojectImageTo3D."""
    def __init__(self, min_disp: int = 0, max_disp: int = 128, block_size: int = 7):
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.block_size = max(3, block_size | 1)  # ensure odd >=3
        self.num_disparities = max_disp - min_disp
        if self.num_disparities % 16 != 0:
            self.num_disparities = ((self.num_disparities // 16) + 1) * 16

    def create_sgbm_matcher(self) -> cv2.StereoSGBM:
        # Use a sensible window and regularization
        block = max(3, self.block_size | 1)  # odd and >=3
        num_disp = self.num_disparities
        return cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=num_disp,
            blockSize=4, #blockSize=block,                 # e.g. 5 or 7
            P1=8  * 3 * block * block,
            P2=32 * 3 * block * block,
            disp12MaxDiff=1,
            uniquenessRatio=7,  #uniquenessRatio=10,              # 10–15 is typical
            speckleWindowSize=80,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )


    def compute_depth_map_with_wls(self, gray_left: np.ndarray, gray_right: np.ndarray,
                                   lambda_val: float = 8000.0, sigma_color: float = 1.5) -> np.ndarray:
        left_matcher = self.create_sgbm_matcher()

        if not hasattr(cv2, "ximgproc"):
            disp = left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
            disp[disp < 0] = 0.0
            print("[WARN] cv2.ximgproc not found; using raw SGBM (no WLS).")
            return disp

        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls.setLambda(lambda_val)
        wls.setSigmaColor(sigma_color)

        dL_q16 = left_matcher.compute(gray_left,  gray_right).astype(np.int16)
        dR_q16 = right_matcher.compute(gray_right, gray_left).astype(np.int16)
        dW_q16 = wls.filter(dL_q16, gray_left, None, dR_q16)

        disp = dW_q16.astype(np.float32) / 16.0  # pixel disparity
        disp[disp < 0] = 0.0
        return disp

# ========== Visualizer ==========
class StereoVisualizer:
    @staticmethod
    def create_rectification_visualization(rect_left: np.ndarray, rect_right: np.ndarray,
                                           scale: float = 0.5, line_spacing: int = 50) -> np.ndarray:
        Ls = cv2.resize(rect_left,  (0, 0), fx=scale, fy=scale)
        Rs = cv2.resize(rect_right, (0, 0), fx=scale, fy=scale)
        pair = np.hstack((Ls, Rs))
        for y in range(0, pair.shape[0], line_spacing):
            cv2.line(pair, (0, y), (pair.shape[1], y), (0, 0, 255), 1)
        return pair

    @staticmethod
    def create_disparity_difference(gray_left: np.ndarray, gray_right: np.ndarray,
                                    target_size: Tuple[int, int] = (854, 480)) -> np.ndarray:
        diff = cv2.absdiff(gray_left, gray_right)
        return cv2.resize(diff, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def normalize_disparity_for_display(disparity: np.ndarray, colormap: int = cv2.COLORMAP_JET):
        disp = np.nan_to_num(disparity, nan=0.0).astype(np.float32)
        disp_u8 = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_col = cv2.applyColorMap(disp_u8, colormap)
        return disp_u8, disp_col

    @staticmethod
    def display_images(images_dict: dict, window_size: Tuple[int, int] = (854, 480)):
        for name, img in images_dict.items():
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, window_size[0], window_size[1])
            cv2.imshow(name, img)

# ========== Point Cloud ==========
class PointCloudGenerator:
    """Reproject using the SAME Q from stereoRectify. Expects float pixel disparity."""
    def __init__(self, Q: np.ndarray):
        self.Q = Q

    def generate_point_cloud(self, disparity: np.ndarray, color_image: np.ndarray,
                             min_disparity_threshold: float = 0.5):
        disp = disparity.astype(np.float32)
        pts3d = cv2.reprojectImageTo3D(disp, self.Q)  # rectified-left frame
        mask = (disp > float(min_disparity_threshold)) & np.isfinite(pts3d).all(axis=2)
        return pts3d[mask], color_image[mask]

    @staticmethod
    def save_point_cloud_ply(filename: str, points: np.ndarray, colors: np.ndarray,
                             header_comment: Optional[str] = None):
        pts = points.reshape(-1, 3).astype(np.float32)
        cols = colors.reshape(-1, 3).astype(np.uint8)
        header = "ply\nformat ascii 1.0\n"
        if header_comment:
            for line in header_comment.splitlines():
                header += f"comment {line}\n"
        header += (
            f"element vertex {len(pts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar blue\nproperty uchar green\nproperty uchar red\n"
            "end_header\n"
        )
        with open(filename, 'w') as f:
            f.write(header)
            for (x, y, z), (b, g, r) in zip(pts, cols):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(b)} {int(g)} {int(r)}\n")

# ========== Pipeline ==========
class StereoVisionPipeline:
    def __init__(self, calibration_file: str):
        self.calib  = StereoCalibrationLoader(calibration_file)
        self.proc   = StereoImageProcessor(self.calib)
        self.depth  = DepthMapGenerator()
        self.vis    = StereoVisualizer()
        self.pcloud = PointCloudGenerator(self.calib.Q)

    def process_stereo_images(self, image_path: str = None, left_path: str = None,
                              right_path: str = None, save_point_cloud: bool = False,
                              point_cloud_filename: str = "output_point_cloud.ply") -> dict:
        # Load
        if image_path:
            L, R = self.proc.load_combined_stereo_image(image_path)
        elif left_path and right_path:
            L, R = self.proc.load_separate_images(left_path, right_path)
        else:
            raise ValueError("Either image_path or both left_path and right_path must be provided")

        # Rectify
        rectL, rectR = self.proc.rectify_images(L, R)

        # Prep for matching
        gL, gR = self.proc.preprocess_for_matching(rectL, rectR, equalize_hist=True)

        # Disparity (float pixels)
        disp = self.depth.compute_depth_map_with_wls(gL, gR)

        # Viz
        rect_vis = self.vis.create_rectification_visualization(rectL, rectR)
        diff_vis = self.vis.create_disparity_difference(gL, gR)
        disp_u8, disp_col = self.vis.normalize_disparity_for_display(disp)

        # Point cloud
        point_cloud_data = None
        if save_point_cloud:
            pts, cols = self.pcloud.generate_point_cloud(disp, rectL)
            self.pcloud.save_point_cloud_ply(point_cloud_filename, pts, cols)
            point_cloud_data = (pts, cols)
            print(f"[OK] Point cloud saved: {point_cloud_filename}  (points: {len(pts)})")

        return {
            "rectified_left": rectL,
            "rectified_right": rectR,
            "gray_left": gL,
            "gray_right": gR,
            "depth_map": disp,  # float32 disparity
            "rectification_visualization": rect_vis,
            "difference_visualization": diff_vis,
            "disparity_grayscale": disp_u8,
            "disparity_colored": disp_col,
            "point_cloud": point_cloud_data
        }

    def display_results(self, results: dict):
        to_show = {
            "Rectified Diff": results["difference_visualization"],
            "Disparity Visual Check": results["disparity_grayscale"],
            "WLS Disparity Map": results["disparity_colored"],
        }
        self.vis.display_images(to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ========== Example ==========
def main():
    try:
        pipeline = StereoVisionPipeline(
            r"C:\Users\sybanzon\Documents\schoolFiles\RESERARCH\full\sept1\opencv_stereo_params2.npz"
        )
        results = pipeline.process_stereo_images(
            image_path=r"C:\Users\sybanzon\Documents\schoolFiles\RESERARCH\calibration\WIN_20250708_14_23_01_Pro.jpg",
            save_point_cloud=True,
            point_cloud_filename=r"C:\Users\sybanzon\Documents\schoolFiles\RESERARCH\full\sept1\output_point_cloud.ply"
        )
        pipeline.display_results(results)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
