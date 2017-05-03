#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

calib = pickle.load(open('calib.p', 'rb'))


def compute_features(image):
    return image

for fname in glob.glob('test_images/*.jpg'):
    image = mpimg.imread(fname)
    features = compute_features(image)
    plt.imshow(features)
    plt.show()
    