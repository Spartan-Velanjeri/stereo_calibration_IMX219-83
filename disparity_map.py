import cv2 as cv
import numpy as np
import math

# Load the stereo map for rectification from the XML file
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

cameraMatrixL = cv_file.getNode('cameraMatrixL').mat()
distL = cv_file.getNode('distL').mat()
cameraMatrixR = cv_file.getNode('cameraMatrixR').mat()
distR = cv_file.getNode('distR').mat()

rot = cv_file.getNode('R').mat()
trans = cv_file.getNode('T').mat()

rectL = cv_file.getNode('rectL').mat()
rectR = cv_file.getNode('rectR').mat()
projMatrixL = cv_file.getNode('projMatrixL').mat()
projMatrixR = cv_file.getNode('projMatrixR').mat()
Q = cv_file.getNode('Q').mat()

cv_file.release()

# Load left and right images for which to compute disparity
imgL = cv.imread('./images/left.jpg')
imgR = cv.imread('./images/right.jpg')

# Apply rectification maps to rectify images
rectifiedL = cv.remap(imgL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4)
rectifiedR = cv.remap(imgR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4)

gray_imageL = cv.cvtColor(rectifiedL, cv.COLOR_BGR2GRAY)
gray_imageR = cv.cvtColor(rectifiedR, cv.COLOR_BGR2GRAY)

# Create a window to adjust parameters
cv.namedWindow('Disparity Map', cv.WINDOW_NORMAL)
cv.namedWindow('Rectified Left', cv.WINDOW_NORMAL)
cv.namedWindow('Rectified Right', cv.WINDOW_NORMAL)
cv.namedWindow('Greyscale Left',cv.WINDOW_NORMAL)
cv.namedWindow('Greyscale Right',cv.WINDOW_NORMAL)
cv.namedWindow('Depth Map', cv.WINDOW_NORMAL)


cv.imshow('Rectified Left', rectifiedL)
cv.imshow('Rectified Right', rectifiedR)
# cv.imshow('Greyscale Left', gray_imageL)
# cv.imshow('Greyscale Right', gray_imageR)

# Trackbars to adjust parameters
def nothing(x):
    pass

cv.createTrackbar('numDisparities', 'Disparity Map', 1, 18, nothing)  # Range [1, 16]
cv.createTrackbar('minDisparity', 'Disparity Map', 1, 16, nothing)
cv.createTrackbar('blockSize', 'Disparity Map', 5, 50, nothing)       # Range [5, 50]
cv.createTrackbar('uniquenessRatio', 'Disparity Map', 10, 100, nothing)  # Range [0, 100]
cv.createTrackbar('speckleWindowSize', 'Disparity Map', 100, 200, nothing)  # Range [0, 200]
cv.createTrackbar('speckleRange', 'Disparity Map', 2, 50, nothing)     # Range [0, 50]

def compute_depth_map(disparity_map, baseline, focal_length_px):
    """
    Calculate depth map from a disparity map.
    
    Parameters:
    - disparity_map (np.ndarray): The disparity map (in pixels).
    - baseline (float): The baseline distance between the cameras (in meters).
    - focal_length_px (float): The focal length in pixels.
    
    Returns:
    - depth_map (np.ndarray): Computed depth map (in meters).
    """
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    valid_disparity = disparity_map > 0
    depth_map[valid_disparity] = (focal_length_px * baseline) / disparity_map[valid_disparity]
    
    return depth_map
def get_depth_from_disparity(disparity_map, x, y, baseline, focal_length_mm, image_width, sensor_width):
    """
    Calculate depth from disparity at a specific coordinate.
    
    Parameters:
    - disparity_map (np.ndarray): The disparity map (in pixels).
    - x, y (int): The coordinates of the pixel to compute the depth for.
    - baseline (float): The baseline distance between the cameras (in meters).
    - focal_length_mm (float): The focal length in millimeters.
    - image_width (int): Width of the image in pixels.
    - sensor_width (float): Width of the camera sensor in millimeters.
    
    Returns:
    - float: Depth at (x, y) in meters.
    """
    # Convert focal length from mm to pixels
    # focal_length_px = (focal_length_mm * image_width) / sensor_width

    # using the projection matrices
    focal_length_px = projMatrixL[0, 0]  # Using the focal length directly from the projection matrix
    # print(focal_length_px)
    
    # Validate coordinates
    if x < 0 or x >= disparity_map.shape[1] or y < 0 or y >= disparity_map.shape[0]:
        print(f"Coordinates ({x}, {y}) are out of bounds.")
        return np.inf
    
    # Get disparity at (x, y)
    disparity = disparity_map[y, x]
    
    # Ensure disparity is positive and non-zero for valid depth calculation
    if disparity > 0:
        depth = (focal_length_px * baseline) / disparity
        return depth
    else:
        return np.inf  # Indicates no depth info if disparity is zero or negative

def focal_length_from_hfov(image_width, hfov, focal_length_mm):
    # Convert horizontal field of view from degrees to radians
    hfov_rad = math.radians(hfov)
    # Calculate focal length in pixels
    focal_length_pix = (image_width / 2) / math.tan(hfov_rad / 2)
    print("Computed focal length in pixels:", focal_length_pix)
    # Use focal_length_mm for informative purposes if needed
    print("Provided focal length in mm:", focal_length_mm)
    return focal_length_pix

# Define baseline and focal length in pixels
baseline = 0.06  # in meters (example: 6 cm baseline)
focal_length_mm = 2.6  # Focal length in mm
sensor_width_mm = 3.2  # Sensor width in mm
hfov = 73.0
image_width_px = imgL.shape[1]  # Image width in pixels
focal_length_px = (focal_length_mm * image_width_px) / sensor_width_mm
focal_length_px = focal_length_from_hfov(image_width_px, hfov, focal_length_mm)

x,y = 1476, 1830


while True:
    # Get current positions of trackbars
    num_disp = cv.getTrackbarPos('numDisparities', 'Disparity Map') * 16  # numDisparities must be divisible by 16
    block_size = cv.getTrackbarPos('blockSize', 'Disparity Map')
    if block_size % 2 == 0:  # blockSize must be odd
        block_size += 1

    uniqueness_ratio = cv.getTrackbarPos('uniquenessRatio', 'Disparity Map')
    speckle_window_size = cv.getTrackbarPos('speckleWindowSize', 'Disparity Map')
    speckle_range = cv.getTrackbarPos('speckleRange', 'Disparity Map')
    minimum_disparity = cv.getTrackbarPos('minDisparity','Disparity Map')
    # Initialize the stereo matcher with updated parameters
    stereo = cv.StereoSGBM_create(
        minDisparity=minimum_disparity,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range
    )

    # Compute the disparity map
    disparity = stereo.compute(rectifiedL, rectifiedR).astype(np.float32) / 16.0
    depth = get_depth_from_disparity(disparity, x, y, baseline, focal_length_mm, image_width_px, sensor_width_mm)
    print(f"Depth at pixel ({x}, {y}) using disparity map: {depth} meters")

    # Reproject disparity to 3D using the Q matrix
    points_3D = cv.reprojectImageTo3D(disparity, Q)
    # Depth map derived from the Z-coordinate
    depth_map_from_Q = points_3D[..., 2]

    # Retrieve depth at specific point
    depth_at_point_from_Q = depth_map_from_Q[y, x]
    print(f"Depth at pixel ({x}, {y}) using Q matrix: {depth_at_point_from_Q:.2f} meters")

    # Normalize the disparity for visualization
    disparity_visual = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
    disparity_visual = np.uint8(disparity_visual)

    # Display the disparity map
    cv.imshow('Disparity Map', disparity_visual)

    # Compute and display the depth map
    depth_map = compute_depth_map(disparity, baseline, focal_length_px)
    depth_map_visual = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX)
    depth_map_visual = np.uint8(depth_map_visual)
    cv.imshow('Depth Map', depth_map_visual)

    # Break the loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv.destroyAllWindows()
