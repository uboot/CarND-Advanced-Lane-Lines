**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./undistorted_chessboard.png "Undistorted"
[image2]: ./undistorted_lines.png "Road Undistorted"
[image3]: ./thresholded.png "Binary Example"
[image4]: ./perspective_transform.png "Road Transformed"
[image5]: ./detected_lines.png "Fit Visual"
[image6]: ./output_images/test1.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 23 through 44 of the file called `camera_calibration.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

Finally, the calibration matrix and the distortion coefficients are stored in the file `calib.p`.

### Pipeline (single images)

The pipeline for single images is contained in `pipeline_images.py`. It imports the calibration from `calib.p` and then uses functions from the module `lane_detector.py` to detect the lane in the images in the directory `test_images`.

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will give a comparison of the image `test_images/straight_lines1.jpg` and its undistored version as computed in lines 53 through 55 in  `camera_calibration.py`.
![alt text][image2]

The actual distortion-correction in the pipeline is computed in line 13 of `lane_detector.py`.

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for computing the perspective transform is found in the lines 59 through 67 in the file `camera_calibration.py`. I chose to extract the source points by manually picking them in the undistorted version of `test_images/straight_lines1.jpg`. Then I chose a square destination region at the center of the target image. This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 235, 690      | 280, 720        | 
| 580, 460      | 280, 0      |
| 705, 460     | 1000, 0      |
| 1070, 690      | 1000, 720        |

The hardcoded arrays are the following:

```python
src = np.float32([[235,690],[580,460],[705,460],[1070,690]])
dst = np.float32([[280,720],[280,0],[1000,0],[1000,720]])
```

I verified that my perspective transform was working as expected by applying the transformation to the image in which I picked the source points:

![alt text][image4]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 15 through 61 in `lane_detector.py`).  Here's an example of my output for this step:
![alt text][image3]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Next, I applied a morphological opening to the binary image in lines 63 through 66 to remove noisy artefacts from the images. Then I fit my lane lines with a 2nd order polynomial in lines 80 through 181.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I computed the radius of the curvature in lines 260 through 274 in my code in `lane_detector.py` and the offset of the vehicle in lines 276 through 287.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 29 through 31 in my code in `pipeline_images.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The computation and selection of the binary features is the most sensitive part the my project code. The lane line fit turned out to be more stable after I added a morphological filtering step which removed small positive regions from the binary image. With this approach I was able to successfully detect the lane in all test images. This is however not the case for test images I took from the challenge videos. For these data the feature selection has to be improved further. I would suggest tuning the parameters of the feature selection systematically using labeled feature data and an automatic approach.

The video pipeline greatly benefits from averaging lane lines detected in previous frames. For all but the first frame I use the function `update_lanes()` which considers previous detection results. However, this approach could be problematic if the result "drifts" away from the correct solution. Because `update_lanes()` searches only in a limited region it can not correctly detect the lane lines if it is initialized with a wrong solution.
To overcome this issue I would add a complete search with the function `find_lanes()` on selected keyframes (e.g. every second).

