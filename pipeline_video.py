#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
    
    if counter%30 == 0:
        left_fit, right_fit, image = ld.find_lanes(warped, left_fits, right_fits)
    else:
        left_fit, right_fit, image = ld.update_lanes(warped, left_fits, right_fits)
        
    if len(left_fits) > 10:
        left_fits.pop(0)
        right_fits.pop(0)
        
    
    unwarped = ld.warp(image, Minv)
    result = ld.compute_overlay(undistorted, unwarped)
    
    curvature = ld.compute_curvature(left_fit, right_fit, image.shape[0])
    offset = ld.compute_offset(left_fit, right_fit, image.shape)
    
    ld.print_data(result, curvature, offset)
    
    counter += 1
    return result


output_file = 'output.mp4'
input_clip = VideoFileClip('project_video.mp4')
result_clip = input_clip.fl_image(process_image) #NOTE: this function expects color images!!
result_clip.write_videofile(output_file, audio=False)