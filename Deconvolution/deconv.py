import cv2
import numpy as np
from scipy.signal import convolve2d, wiener
from skimage import restoration
from skimage.restoration import unsupervised_wiener

def blind_deconvolution(image, init_psf):
    deconvolved, psf = unsupervised_wiener(image, init_psf)
    return deconvolved

def richardson_lucy_deconvolution(image, psf):
    deconvolved = restoration.richardson_lucy(image, psf,num_iter=30)
    return deconvolved

def wiener_deconvolution(img, kernel, K):
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.fft.ifft2(dummy)
    dummy = np.abs(dummy)
    return dummy

image = cv2.imread('motionblurCar.jpg', cv2.IMREAD_GRAYSCALE)
init_psf = np.array([[0, 0, 1, 0, 0],
                     [0, 1, 2, 1, 0],
                     [1, 2, 3, 2, 1],
                     [0, 1, 2, 1, 0],
                     [0, 0, 1, 0, 0]], np.float32) / 5
wiener_deconvolved_image = wiener_deconvolution(image, init_psf, 0.01)
deconvolved_image = richardson_lucy_deconvolution(image, init_psf)
blind_deconvolved_image = blind_deconvolution(image, init_psf)
cv2.imshow('Deconvolved Image', blind_deconvolved_image)
cv2.imshow('Wiener Deconvolved Image', wiener_deconvolved_image)
cv2.imshow('Richardson Lucy Deconvolved Image', deconvolved_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
