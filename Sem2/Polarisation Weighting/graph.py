#opencv import
import cv2
import numpy as np
import time
import os
import sys
import math
import random
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlate2d
from scipy.optimize import minimize
import scipy.interpolate
from skimage.restoration import wiener
from numpy.linalg import lstsq
import skimage
from scipy.ndimage import convolve
import seaborn as sns
from scipy.fftpack import fft2, ifft2, fftshift
import concurrent.futures
from scipy.ndimage import shift as nd_shift
from scipy.signal import find_peaks

# image_path = "matlab/blueblack.jpeg"
image_path = "rubix/nopol.jpg"
# image_path = "matlab/flower.jpg"
# original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# read rgb
original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
# resize image to 30% of original size
original_image = cv2.resize(original_image, (0,0), fx=0.15, fy=0.15)
# get image size
height, width = original_image.shape[:2]
image_size = (height, width)
I1 = original_image.copy()
I2 = I1.copy()

shift = 15
w1 = 0.3
w2 = 0.7

# I2[:,shift:] = I2[:,:-shift]
I2[:, :-shift] = I2[:, shift:]
# if not greyscale image go int
if len(I1.shape) > 2:
    It = (w1 * I1.astype(int) + w2 * I2.astype(int)).astype(int)


# crop each edge by shift
# It = It[:, shift:-shift]

import mplcursors

def cross_correlation_channel_fft(It_channel, max_shift=20):
    shift_values = []
    correlation_values = []

    for x_shift in range(-max_shift, max_shift + 1):
        # Apply shift to the image channel along the x-axis
        It_channel_shifted = nd_shift(It_channel, shift=(0, x_shift), mode='constant', cval=0)

        # Compute FFT of both original and shifted images
        f1 = fft2(It_channel)
        f2 = fft2(It_channel_shifted)

        # Compute cross-correlation
        cross_corr = np.real(ifft2(f1 * np.conj(f2)))
        cross_corr = fftshift(cross_corr)  # Center the zero-shift at the middle

        # Find the correlation value at the middle (zero shift)
        middle_idx = cross_corr.shape[1] // 2
        correlation_value = cross_corr[:, middle_idx].max()

        # Store the shift and its corresponding correlation value
        shift_values.append(x_shift)
        correlation_values.append(correlation_value)

    # Convert to numpy arrays for easier handling
    shift_values = np.array(shift_values)
    correlation_values = np.array(correlation_values)

    # Find peaks in the correlation values
    peaks, _ = find_peaks(correlation_values)

    # Sort the peaks by their correlation values in descending order
    sorted_peaks = peaks[np.argsort(correlation_values[peaks])[::-1]]

    # Visualization of the peaks

    plt.figure()
    plt.plot(shift_values, correlation_values, label="Cross-Correlation")
    plt.title("Cross-Correlation Peaks")
    plt.xlabel("Shift (pixels)")
    plt.ylabel("Correlation Value")
    plt.grid(True)

    # # Label each x point with its y value
    # for x, y in zip(shift_values, correlation_values):
    #     plt.text(x, y, f"({y:.2e})", ha='center', va='bottom', fontsize=8)

    # Enable interactive cursor
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.2f}, {sel.target[1]:.2e})"))

    plt.show()

# Separate the RGB channels
It_r, It_g, It_b = It[:, :, 0], It[:, :, 1], It[:, :, 2]

# Process each channel sequentially
cross_correlation_channel_fft(It_r)
cross_correlation_channel_fft(It_g)
cross_correlation_channel_fft(It_b)