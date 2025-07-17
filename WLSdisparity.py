import numpy as np
import cv2

# Load calibration and rectification parameters
params = np.load("opencv_stereo_params.npz")

# Load rectification maps and Q
# Load stereo calibration parameters
params = np.load("opencv_stereo_params.npz")
camera_matrix_1 = params['camera_matrix_1']
dist_coeffs_1 = params['dist_coeffs_1']
camera_matrix_2 = params['camera_matrix_2']
dist_coeffs_2 = params['dist_coeffs_2']
R1 = params['R1']
R2 = params['R2']
P1 = params['P1']
P2 = params['P2']
Q = params['Q']

image_size = tuple(params['image_size'])  # Should be (width, height)

# Regenerate rectification maps from scratch
map1x, map1y = cv2.initUndistortRectifyMap(
    camera_matrix_1, dist_coeffs_1, R1, P1, image_size, cv2.CV_32FC1)

map2x, map2y = cv2.initUndistortRectifyMap(
    camera_matrix_2, dist_coeffs_2, R2, P2, image_size, cv2.CV_32FC1)


# Load the combined stereo image (3840x1080)
img_combined = cv2.imread(r"C:\Users\pvbas\Music\secret files\WIN_20250708_14_23_01_Pro.jpg")
if img_combined is None:
    raise FileNotFoundError("Combined stereo image not found.")

# Split into left and right images
h, w = img_combined.shape[:2]
img_left = img_combined[:, :w//2]
img_right = img_combined[:, w//2:]

# # # Use this block if the images are already separate
# img_left = cv2.imread(r"C:\Users\pvbas\Music\secret files\scene1.row3.col3.png")
# img_right = cv2.imread(r"C:\Users\pvbas\Music\secret files\scene1.row3.col1.png")

# Rectify images using the maps
rect_left = cv2.remap(img_left, map1x, map1y, interpolation=cv2.INTER_LINEAR)
rect_right = cv2.remap(img_right, map2x, map2y, interpolation=cv2.INTER_LINEAR)

# Resize for display (optional, adjust scale as needed)
display_scale = 0.5
rect_left_small = cv2.resize(rect_left, (0, 0), fx=display_scale, fy=display_scale)
rect_right_small = cv2.resize(rect_right, (0, 0), fx=display_scale, fy=display_scale)

# Stack resized images side by side
rect_pair = np.hstack((rect_left_small, rect_right_small))

# Draw horizontal lines every 50 pixels
line_color = (0, 0, 255)  # Red in BGR
thickness = 1
height = rect_pair.shape[0]
for y in range(0, height, 50):
    cv2.line(rect_pair, (0, y), (rect_pair.shape[1], y), line_color, thickness)

# Convert to grayscale for block matching
gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

gray_left = cv2.equalizeHist(gray_left)
gray_right = cv2.equalizeHist(gray_right)

# Overlay the two images 
rectified_diff = cv2.absdiff(gray_left, gray_right)
resized_diff = cv2.resize(rectified_diff, (854, 480), interpolation=cv2.INTER_AREA)
cv2.namedWindow("Rectified Diff", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rectified Diff", 854, 480)
cv2.imshow("Rectified Diff", resized_diff)

# Replace your StereoBM block with this:
min_disp = 0
max_disp = 128  # SGM constraint, must be divisible by 16
num_disparities = max_disp - min_disp  # 128

if num_disparities % 16 != 0:
    num_disparities = ((num_disparities // 16) + 1) * 16

block_size = 7 # Try 5, 7, 9, 11, etc.

# Try different blockSize (odd numbers: 3, 5, 7, 9, 11, 15)
# Try different numDisparities (must be divisible by 16)

left_matcher = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Create right matcher automatically
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# Create WLS filter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000)  # Try 8000 to 15000
wls_filter.setSigmaColor(1.5)  # Try 0.8 to 2.0

# Compute disparities
disparity_left = left_matcher.compute(gray_left, gray_right).astype(np.int16)
disparity_right = right_matcher.compute(gray_right, gray_left).astype(np.int16)

# Apply WLS filter
disparity_wls = wls_filter.filter(disparity_left, gray_left, None, disparity_right)

# Mask invalid disparities
#disparity[disparity < min_disp] = np.nan

# ---- Choose display mode (SGM): comment/uncomment as needed ----

# # --- Colored disparity map (default) ---
# disparity_clean = np.nan_to_num(disparity, nan=0)
# disp_display = cv2.normalize(disparity_clean, None, 0, 255, cv2.NORM_MINMAX)
# disp_display = np.uint8(disp_display)
# #disp_display = cv2.medianBlur(disp_display, 5)  # Try 3 or 5   //median blur for better image
# disp_color = cv2.applyColorMap(disp_display, cv2.COLORMAP_JET)
# cv2.namedWindow("Disparity Map (SGM/Jet)", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Disparity Map (SGM/Jet)", 854,480)
# cv2.imshow("Disparity Map (SGM/Jet)", disp_color)

# # --- Greyscale disparity map (corrected version) ---
# disparity_clean_gray = np.nan_to_num(disparity, nan=0)
# disp_display_gray = cv2.normalize(disparity_clean_gray, None, 0, 255, cv2.NORM_MINMAX)
# disp_display_gray = np.uint8(disp_display_gray)
# cv2.namedWindow("Disparity Map (SGM/Gray)", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Disparity Map (SGM/Gray)", disp_display_gray.shape[1], disp_display_gray.shape[0])
# cv2.imshow("Disparity Map (SGM/Gray)", disp_display_gray)

# Normalize WLS output for display
disp_wls_normalized = cv2.normalize(disparity_wls, None, 0, 255, cv2.NORM_MINMAX)
disp_wls_normalized = np.uint8(disp_wls_normalized)
disp_wls_color = cv2.applyColorMap(disp_wls_normalized, cv2.COLORMAP_JET)

cv2.namedWindow("WLS Disparity Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("WLS Disparity Map", 854, 480)
cv2.imshow("WLS Disparity Map", disp_wls_color)


cv2.waitKey(0)
cv2.destroyAllWindows()
