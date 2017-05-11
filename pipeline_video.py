#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import pickle
from moviepy.editor import VideoFileClip

import lane_detector as ld

calib = pickle.load(open('calib.p', 'rb'))
mtx = calib['mtx']
dist = calib['dist']
M = calib['M']
Minv = calib['Minv']

def process_image(image):
    undistorted = ld.undistort(image, mtx, dist)
    features = ld.combined_thresh(undistorted)
    warped = ld.warp(features, M)
    
    left_fit, right_fit, image = ld.find_lanes(warped)
    
    left_fit, right_fit, image = ld.update_lanes(warped, left_fit, right_fit)
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
    return result


output_file = 'output.mp4'
input_clip = VideoFileClip("project_video.mp4")
result_clip = input_clip.fl_image(process_image) #NOTE: this function expects color images!!
result_clip.write_videofile(output_file, audio=False)