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

cv.imwrite('grey_L.jpg', gray_imageL)
cv.imwrite('grey_R.jpg', gray_imageR)

def find_depth(right_point, left_point, frame_right, frame_left, baseline,f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right= frame_right.shape
    height_left, width_left= frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point[0]
    x_left = left_point[0]

    # CALCULATE THE DISPARITY:
    disparity = x_left-x_right      #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity             #Depth in [cm]

    return zDepth

# Choosing the A of Voyage as the point
# Left Point: 1734, 1854
# Right Point: 1732, 1854
# img1_rectified, img2_rectified

right_point = (1579, 1860)
left_point = (1510, 1860)
baseline = 6 # in mm
f = 2.6 # in mm
alpha = 73 # in degrees

zDepth = find_depth(right_point, left_point, gray_imageR, gray_imageL, baseline,f, alpha)
print(f"Depth of the box is around : {zDepth/100} metres" ) # -1.92 Metres