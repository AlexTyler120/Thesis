import numpy as np
from scipy import signal
from skimage import io, color
import matplotlib.pyplot as plt
import cv2

# Step 1: Load and convert the image to grayscale
image = io.imread('python/flower.jpg')
if image.ndim == 3:
    image = color.rgb2gray(image)

#resize image 0.15
image = cv2.resize(image, (0,0), fx=0.15, fy=0.15)
# Step 2: Compute the autocorrelation using correlate2d
autocorrelation = signal.correlate2d(image, image, boundary='symm', mode='same')

# Step 3: Normalize the autocorrelation
autocorrelation /= autocorrelation.max()

# Step 4: Extract a 1D slice from the autocorrelation
# For example, take the middle row or column
mid_row = autocorrelation[autocorrelation.shape[0] // 2, :]
mid_col = autocorrelation[:, autocorrelation.shape[1] // 2]

# Step 5: Plot the autocorrelation values
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot along the middle row
ax[0].plot(mid_row)
ax[0].set_title('Autocorrelation (Middle Row)')
ax[0].set_xlabel('Lag')
ax[0].set_ylabel('Autocorrelation Value')

# Plot along the middle column
ax[1].plot(mid_col)
ax[1].set_title('Autocorrelation (Middle Column)')
ax[1].set_xlabel('Lag')
ax[1].set_ylabel('Autocorrelation Value')


image = 0.3 * image[:, 10:] + 0.7 * image[:, :-10]

# Step 2: Compute the autocorrelation using correlate2d
autocorrelation = signal.correlate2d(image, image, boundary='symm', mode='same')

# Step 3: Normalize the autocorrelation
autocorrelation /= autocorrelation.max()

# Step 4: Extract a 1D slice from the autocorrelation
# For example, take the middle row or column
mid_row = autocorrelation[autocorrelation.shape[0] // 2, :]
mid_col = autocorrelation[:, autocorrelation.shape[1] // 2]

# Step 5: Plot the autocorrelation values
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot along the middle row
ax[0].plot(mid_row)
ax[0].set_title('Autocorrelation (Middle Row)')
ax[0].set_xlabel('Lag')
ax[0].set_ylabel('Autocorrelation Value')

# Plot along the middle column
ax[1].plot(mid_col)
ax[1].set_title('Autocorrelation (Middle Column)')
ax[1].set_xlabel('Lag')
ax[1].set_ylabel('Autocorrelation Value')

plt.show()
