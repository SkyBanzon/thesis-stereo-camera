import numpy as np
import scipy.io as sio
import cv2

def matlab_to_opencv_stereo_params(mat_file_path):
    mat_data = sio.loadmat(mat_file_path)
    print("Top-level keys in .mat file:")
    for key in mat_data:
        if not key.startswith('__'):
            print(f"  {key}: {type(mat_data[key])}")

    if 'stereoParamsStruct' not in mat_data:
        raise ValueError("stereoParamsStruct not found in .mat file.")

    params = mat_data['stereoParamsStruct'][0, 0]
    cam1 = params['CameraParameters1'][0, 0]
    cam2 = params['CameraParameters2'][0, 0]

    # Intrinsic matrices
    K1 = cam1['K']
    K2 = cam2['K']

    # Distortion coefficients (flatten and concatenate radial + tangential)
    D1 = np.concatenate([cam1['RadialDistortion'].flatten(), cam1['TangentialDistortion'].flatten()])
    D2 = np.concatenate([cam2['RadialDistortion'].flatten(), cam2['TangentialDistortion'].flatten()])

    # Rotation and translation
    R = params['RotationOfCamera2']
    T = params['TranslationOfCamera2']

    # Image size (height, width)
    image_size = tuple(int(x) for x in cam1['ImageSize'].flatten())

    # Optionally extract more fields
    version = params['Version'] if 'Version' in params.dtype.fields else None
    rect_params = params['RectificationParams'] if 'RectificationParams' in params.dtype.fields else None

    opencv_params = {
        'camera_matrix_1': K1,
        'dist_coeffs_1': D1,
        'camera_matrix_2': K2,
        'dist_coeffs_2': D2,
        'R': R,
        'T': T,
        'image_size': (image_size[1], image_size[0]),  # OpenCV expects (width, height)
        'version': version,
        'rectification_params': rect_params
    }
    return opencv_params

def compute_stereo_rectification(opencv_params):
    """
    Compute stereo rectification parameters using OpenCV.
    
    Args:
        opencv_params (dict): OpenCV stereo calibration parameters
        
    Returns:
        dict: Rectification parameters
    """
    
    K1 = np.asarray(opencv_params['camera_matrix_1'], dtype=np.float64)
    D1 = np.asarray(opencv_params['dist_coeffs_1'], dtype=np.float64).flatten()
    K2 = np.asarray(opencv_params['camera_matrix_2'], dtype=np.float64)
    D2 = np.asarray(opencv_params['dist_coeffs_2'], dtype=np.float64).flatten()
    R = np.asarray(opencv_params['R'], dtype=np.float64)
    T = np.asarray(opencv_params['T'], dtype=np.float64).flatten()
    image_size = tuple(map(int, opencv_params['image_size']))
    
    # Compute rectification transforms
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, 
        image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0  # 0 = maximize image area, 1 = retain all pixels
    )
    
    # Compute rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    
    rectification_params = {
        'R1': R1,  # Rectification rotation for camera 1
        'R2': R2,  # Rectification rotation for camera 2
        'P1': P1,  # Projection matrix for camera 1
        'P2': P2,  # Projection matrix for camera 2
        'Q': Q,    # Disparity-to-depth mapping matrix
        'roi1': roi1,  # Region of interest for camera 1
        'roi2': roi2,  # Region of interest for camera 2
        'map1x': map1x, 'map1y': map1y,  # Rectification maps for camera 1
        'map2x': map2x, 'map2y': map2y   # Rectification maps for camera 2
    }
    
    return rectification_params

def save_opencv_params(opencv_params, rectification_params, output_file):
    """
    Save OpenCV parameters to a file.
    
    Args:
        opencv_params (dict): OpenCV stereo calibration parameters
        rectification_params (dict): Rectification parameters
        output_file (str): Output file path (.npz format)
    """
    
    # Combine all parameters
    all_params = {**opencv_params, **rectification_params}
    
    # Save to .npz file
    np.savez(output_file, **all_params)
    print(f"OpenCV parameters saved to {output_file}")

# Example usage
if __name__ == "__main__":
    mat_file_path = "stereoParamsStruct.mat"
    opencv_params = matlab_to_opencv_stereo_params(mat_file_path)
    print("OpenCV parameters:")
    for k, v in opencv_params.items():
        print(f"{k}: {v}")

    # Optionally compute rectification and save
    rectification_params = compute_stereo_rectification(opencv_params)
    save_opencv_params(opencv_params, rectification_params, "opencv_stereo_params.npz")


# Alternative function for different MATLAB stereo parameter structures
def matlab_to_opencv_alternative(mat_file_path):
    """
    Alternative conversion function for different MATLAB stereo parameter formats.
    """
    
    mat_data = sio.loadmat(mat_file_path)
    
    # Print available fields to help identify the correct structure
    print("Available fields in .mat file:")
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {type(mat_data[key])}")
    
    # You may need to adjust these field names based on your specific .mat file
    # Common alternatives:
    # - stereoParams vs stereoParameters
    # - CameraParameters1/2 vs Camera1/2
    # - IntrinsicMatrix vs cameraMatrix
    # - RadialDistortion vs distortionCoefficients
    
    return mat_data
