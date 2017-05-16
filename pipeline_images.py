#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import lane_detector as ld

calib = pickle.load(open('calib.p', 'rb'))
mtx = calib['mtx']
dist = calib['dist']
M = calib['M']
Minv = calib['Minv']
    
for fname in glob.glob('test_images/*.jpg'):
    image = mpimg.imread(fname)
    undistorted = ld.undistort(image, mtx, dist)
    features = ld.combined_thresh(undistorted)
    features = ld.morph_filter(features)
    warped = ld.warp(features, M)
    
    left_fit, right_fit, image = ld.find_lanes(warped)
    
    left_fit, right_fit, image = ld.update_lanes(warped, [left_fit], [right_fit])
    unwarped = ld.warp(image, Minv)
    result = ld.compute_overlay(undistorted, unwarped)
    
    curvature = ld.compute_curvature(left_fit, right_fit, image.shape[0])
    offset = ld.compute_offset(left_fit, right_fit, image.shape)
    ld.print_data(result, curvature, offset)
    
    plt.imshow(result)
    plt.show()

    