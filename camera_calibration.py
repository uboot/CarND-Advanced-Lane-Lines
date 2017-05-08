#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import pickle
import cv2
import glob
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_images(img, undist):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    #ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undist)
    #ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


nx = 9
ny = 6

objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

objpoints = []
imgpoints = []
for fname in glob.glob('camera_cal/calibration*.jpg'):
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = mpimg.imread('camera_cal/calibration1.jpg')
undist = cv2.undistort(img, mtx, dist, None, mtx)
plot_images(img, undist)

img = mpimg.imread('test_images/straight_lines1.jpg')
undist = cv2.undistort(img, mtx, dist)
plot_images(img, undist)
mpimg.imsave('undistorted.jpg', undist)

img = undist
src = np.float32([[190,720],[580,460],[705,460],[1115,720]])
dst = np.float32([[280,720],[280,0],[1000,0],[1000,720]])
M = cv2.getPerspectiveTransform(src, dst)

calib = {
    'mtx': mtx,
    'dist': dist,
    'M': M
}
pickle.dump(calib, open('calib.p', 'wb'))

warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
plot_images(undist, warped)


