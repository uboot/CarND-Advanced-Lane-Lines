#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import pickle
import cv2
import glob
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# plots two images next to each other
def plot_images(img1, img2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

# pattern size
nx = 9
ny = 6

# coordinates of the pattern points
objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

# try to detect a chessboard on each calibration image
objpoints = []
imgpoints = []
for fname in glob.glob('camera_cal/calibration*.jpg'):
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # if successful append the detected corners and a copy of the pattern points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        
# calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# for illustration read on of the images and display it together with its 
# undistorted version
img = mpimg.imread('camera_cal/calibration1.jpg')
undist = cv2.undistort(img, mtx, dist, None, mtx)
plot_images(img, undist)

# then store an undistorted version of an images with straight lane
img = mpimg.imread('test_images/straight_lines1.jpg')
undist = cv2.undistort(img, mtx, dist)
plot_images(img, undist)

img = undist

# these coordinates correspond to the lane trapezoid in the undistorted image
src = np.float32([[235,690],[580,460],[705,460],[1070,690]])

# we want to map them to this square
dst = np.float32([[280,720],[280,0],[1000,0],[1000,720]])

# compute the transform and its inverse
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# save this data
calib = {
    'mtx': mtx,
    'dist': dist,
    'M': M,
    'Minv': Minv
}
pickle.dump(calib, open('calib.p', 'wb'))

# again plot two images to illustrate the effect of the perspective transform
warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
plot_images(undist, warped)


