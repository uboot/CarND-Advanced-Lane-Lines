#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import matplotlib.pyplot as plt
import numpy as np

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist)

def combined_thresh(img, plot=False):
    # convert the image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # compute the absolute derivative in x of the gray scale image scale it to [0, 255]
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # threshold the x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # convert to HLS color space and extract the L and S channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # threshold the S channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # threshold the L channel
    l_thresh_min = 20
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1
    
    # Combine the three binary thresholds. We activate the image regions with
    # responses in both the S and L channels and all regions with the correct
    # x gradient
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (l_binary == 1)) | (sxbinary == 1)] = 1
    
    if plot:
        # stack the thresholded channels (gradient, S and L) for visualization
        color_binary = 255*np.dstack((l_binary, sxbinary, s_binary)) 
        plt.imshow(color_binary)
        plt.show()    
        plt.imshow(combined_binary, cmap='gray')
        plt.show()    
    
    return combined_binary

def morph_filter(img, plot=False):
    # remove small structures from the binary input image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    filtered = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if plot:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img, cmap='gray')
        ax2.imshow(filtered, cmap='gray')
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()  
        
    return filtered

def warp(img, M):
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
 
def find_lanes(binary_warped, plot=False):
    """
    Find lane lines in a warped binary image. Returns the coefficients of
    polynomial fitting the left and right lane lines together with a color image.
    The image contains visualizations of the left (red) and right (blue) lane 
    lines and the lane region (green).
    """
    
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    bottom_half_height = np.int(binary_warped.shape[0]/2)
    histogram = np.sum(binary_warped[bottom_half_height:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Compute a dense discretization of the left and right lane line
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Draw the green lane onto the warped blank image
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line = np.array([np.transpose(np.vstack([right_fitx[::-1], ploty[::-1]]))])
    lane_pts = np.hstack((left_line, right_line))
    cv2.fillPoly(out_img, np.int_([lane_pts]), (0,255, 0))
    
    # Draw the left and right lane regions as red and blue
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    if plot:
        # Optionally, plot the lane line fit as yellow lines
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    
    return left_fit, right_fit, out_img

def update_lanes(binary_warped, left_fits=[], right_fits=[], plot=False):
    """
    Find lane lines in a warped binary image considering the results of 
    previous fits. This functions limits the search for lane lines to the region
    around the means of the input lines left_fits and right_fits.
    
    Returns the coefficients of polynomial fitting the left and right lane lines 
    together with a color image. The image contains visualizations of the left
    (red) and right (blue) lane lines and the lane region (green).
    """
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_fit = np.mean(np.array(left_fits), axis=0)
    right_fit = np.mean(np.array(right_fits), axis=0)
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fits.append(np.polyfit(lefty, leftx, 2))
    right_fits.append(np.polyfit(righty, rightx, 2))
    left_fit = np.mean(np.array(left_fits), axis=0)
    right_fit = np.mean(np.array(right_fits), axis=0)
    
    # Compute a dense discretization of the left and right lane line
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    
    # Draw the lane onto the warped blank image
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line = np.array([np.transpose(np.vstack([right_fitx[::-1], ploty[::-1]]))])
    lane_pts = np.hstack((left_line, right_line))
    cv2.fillPoly(out_img, np.int_([lane_pts]), (0,255, 0))
    
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    if plot:
        # Optionally, plot the lane line fit as yellow lines
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    
    return left_fit, right_fit, out_img

def convert_to_meters(fit):
    # Define conversions in x and y from pixels space to meters    
    return xm_per_pix * fit/np.array([ym_per_pix**2, ym_per_pix, 1])
    
def compute_curvature(left_fit, right_fit, y_eval, verbose = False):   
    # Compute polynomials in world space
    left_fit_cr = convert_to_meters(left_fit)
    right_fit_cr = convert_to_meters(right_fit)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters
    if verbose:
        print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    
    return (left_curverad + right_curverad) / 2

def compute_offset(left_fit, right_fit, image_shape, verbose = False):
    y_eval = image_shape[0]
    left_fit_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_fit_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    center_x = (left_fit_x+right_fit_x)/2
    offset = center_x - image_shape[1]/2
    offset *= xm_per_pix
    
    if verbose:
        print(offset, 'm')
    
    return offset

def print_data(image, curvature, offset): 
    curvature_string = 'curvature = {0:.0f} m'.format(curvature)
    offset_string = 'offset = {0:1.2f} m'.format(offset)
    cv2.putText(image, curvature_string, (50, 80), cv2.FONT_HERSHEY_PLAIN, 4,
                (255, 0, 255), thickness=3)
    cv2.putText(image, offset_string, (50, 140), cv2.FONT_HERSHEY_PLAIN, 4,
                (255, 0, 255), thickness=3)

def compute_overlay(background, overlay):
    inverse = 255 - overlay
    white = np.logical_and.reduce(inverse[:,:,:]==[255,255,255], axis=2)
    inverse[white] = [0, 0, 0]
    return cv2.addWeighted(background, 1.0, inverse, -0.3, 0)
    