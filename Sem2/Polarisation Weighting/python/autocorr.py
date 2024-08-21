import numpy as np
from scipy import signal
from skimage import io, color
import matplotlib.pyplot as plt
import cv2
from scipy.fft import fft2, ifft2, fftshift

# Step 1: Load and convert the image to grayscale
image = io.imread('python/flower.jpg')
if image.ndim == 3:
    image = color.rgb2gray(image)

#resize image 0.15
image = cv2.resize(image, (0,0), fx=0.05, fy=0.05)

# Compute autocorrelation using signal.correlate2d
autocorr1 = signal.correlate2d(image, image, mode='full')
print(autocorr1)

# Pad the image to match the output size of correlate2d
pad_size = (image.shape[0] - 1, image.shape[1] - 1)
padded_image = np.pad(image, [pad_size, pad_size], mode='constant')

# Compute the Fourier transform of the padded image
image_fft = fft2(padded_image)

# Compute the power spectrum (element-wise product of the FFT with its complex conjugate)
power_spectrum = image_fft * np.conj(image_fft)

# Compute the inverse FFT to get the autocorrelation
autocorr = ifft2(power_spectrum).real

# Trim the autocorrelation result to match (2M-1, 2N-1) size
original_shape = image.shape
autocorr = autocorr[:(2*original_shape[0]-1), :(2*original_shape[1]-1)]

print(autocorr)

# Compare the results with a higher tolerance
print(np.allclose(autocorr1, autocorr, atol=1e-4, rtol=1e-4))