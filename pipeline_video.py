#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import pickle
from moviepy.editor import VideoFileClip

import lane_detector as ld

calib = pickle.load(open('calib.p', 'rb'))
mtx = calib['mtx']
dist = calib['dist']
M = calib['M']
Minv = calib['Minv']

left_fits = []
right_fits = []
counter = 0

def process_image(image):
    global left_fits, right_fits, counter
    
    undistorted = ld.undistort(image, mtx, dist)
    features = ld.combined_thresh(undistorted)
    warped = ld.warp(features, M)
    
    if counter == 0:
        left_fit, right_fit, image = ld.find_lanes(warped)
        left_fits = [left_fit]
        right_fits = [right_fit]
    else:
        left_fit_mean = np.mean(np.array(left_fits), axis=0)
        right_fit_mean = np.mean(np.array(right_fits), axis=0)
        left_fit, right_fit, image = ld.update_lanes(warped, left_fit_mean, right_fit_mean)
        left_fits.append(left_fit)
        right_fits.append(right_fit)
        if len(left_fits) > 5:
            left_fits.pop(0)
            right_fits.pop(0)
        
    unwarped = ld.warp(image, Minv)
    
    result = cv2.addWeighted(undistorted, 1.0, unwarped, 0.3, 0)
    
    curvature = ld.compute_curvature(left_fit, right_fit, image.shape[0])
    offset = ld.compute_offset(left_fit, right_fit, image.shape)
    
    curvature_string = 'curvature = {0:.0f} m'.format(curvature)
    offset_string = 'offset = {0:1.2f} m'.format(offset)
    cv2.putText(result, curvature_string, (50, 80), cv2.FONT_HERSHEY_PLAIN, 4,
                (255, 0, 255), thickness=3)
    cv2.putText(result, offset_string, (50, 140), cv2.FONT_HERSHEY_PLAIN, 4,
                (255, 0, 255), thickness=3)
    
    counter += 1
    return result


output_file = 'output.mp4'
input_clip = VideoFileClip('project_video.mp4')
result_clip = input_clip.fl_image(process_image) #NOTE: this function expects color images!!
result_clip.write_videofile(output_file, audio=False)