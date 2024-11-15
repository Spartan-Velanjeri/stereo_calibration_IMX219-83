# stereo_calibration_IMX219-83

## Setup (in Raspberry Pi)
1. Add overlay in /boot/firmware/config.txt (Add the below two lines at the end of file in seperate lines)
``
dtoverlay=imx219,cam0                    
dtoverlay=imx219,cam1
``
2. Change camera_auto_detect=1 to camera_auto_detect=0
3. pip install opencv-contrib-python

## To Run
1. To test individual cameras (Here running camera connected to port 0):
``
libcamera-hello --camera 0 -t 0
``
2. To run manual calibration, run ``stereo_calibration.py``. Make sure to change the chessboard dimensions, square size and the path to the calibration images before you do the calibration. The result of the program will a stereoMap.xml file which stores all calibration data.

3. Using the stereoMap.xml, you can run the ``disparity_map.py``. This allows you to use the StereoSGBM algorithm to generate the disparity map. The workflow of the program: Rectification --> Disparity with StereoSGBM (Tweakable params) --> Depth at a point with the Q Matrix a --> Depth at a point using the disparity, baseline and focal length.

    Parameters to use:

    numDisparities: 06/16

    blockSize: 06/50

    uniquenessRatio: 10/100

    speckleWindowSize: 182/200

    speckleRange: 02/50

4. Using the stereoMap.xml, you can run the ``triangulation.py``. This program helps you get the depth at a particular object without the disparity map. It works by calculating the disparity of a object by difference between the x coordinates of the same object in both the images (Hence rectification is crucial)

Calibration related images aren't attached here but can be requested by contacting me :)