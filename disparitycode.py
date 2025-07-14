import numpy as np
import cv2

# Load calibration and rectification parameters
params = np.load("opencv_stereo_params.npz")

# Load rectification maps and Q
map1x = params['map1x']
map1y = params['map1y']
map2x = params['map2x']
map2y = params['map2y']
Q = params['Q']

# Load the combined stereo image (3840x1080)
img_combined = cv2.imread(r"WIN_20250708_14_23_01_Pro.jpg")
if img_combined is None:
    raise FileNotFoundError("Combined stereo image not found.")

# Split into left and right images
h, w = img_combined.shape[:2]
img_left = img_combined[:, :w//2]
img_right = img_combined[:, w//2:]

# Use this block if the images are already separate
# img_left = cv2.imread(r"C:\Users\sybanzon\Downloads\aloeL.jpg")
# img_right = cv2.imread(r"C:\Users\sybanzon\Downloads\aloeR.jpg")

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

# Replace your StereoBM block with this:
min_disp = 0
max_disp = 128  # SGM constraint, must be divisible by 16
num_disparities = max_disp - min_disp  # 128

if num_disparities % 16 != 0:
    num_disparities = ((num_disparities // 16) + 1) * 16

block_size = 3 # Try 5, 7, 9, 11, etc.

# Try different blockSize (odd numbers: 3, 5, 7, 9, 11, 15)
# Try different numDisparities (must be divisible by 16)

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=5,  # Try lowering to 5-8
    speckleWindowSize=100,  # May increase to reduce background speckle
    speckleRange=32,    # Try smaller range for smoother output
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute disparity map
disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

# Mask invalid disparities
disparity[disparity < min_disp] = np.nan

# ---- Choose display mode: comment/uncomment as needed ----

# --- Colored disparity map (default) ---
disp_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_display = np.uint8(np.nan_to_num(disp_display, nan=0))
disp_color = cv2.applyColorMap(disp_display, cv2.COLORMAP_JET)
cv2.namedWindow("Disparity Map (SGM/Jet)", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Disparity Map (SGM/Jet)", disp_color.shape[1], disp_color.shape[0])
cv2.resizeWindow("Disparity Map (SGM/Jet)", 854,480)
cv2.imshow("Disparity Map (SGM/Jet)", disp_color)

# #--- Greyscale disparity map (uncomment to use) ---
# disp_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
# disp_display = np.uint8(np.nan_to_num(disp_display, nan=0))
# cv2.namedWindow("Disparity Map (SGM/Gray)", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Disparity Map (SGM/Gray)", disp_display.shape[1], disp_display.shape[0])
# cv2.imshow("Disparity Map (SGM/Gray)", disp_display)

cv2.waitKey(0)
cv2.destroyAllWindows()
