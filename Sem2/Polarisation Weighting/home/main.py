import ImageRun
import PatchRun
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import PatchGetAndCombine
from matplotlib.colors import LogNorm
import WeightingEstimate
import Autocorrelation
from mpl_toolkits.mplot3d import Axes3D
from gt import gt

def main():
    RESIZE_VAR = 1
    GREY = False
    SIMULATED_SHIFT = 10
    WEIGHTING_SIM = 0.6
    ANGLE = 0
    PATCH_SIZE = 24

    ### Shift estimates with polarised images ###
    transformed_image = ImageRun.polarised_generation("dragon", ANGLE, RESIZE_VAR, GREY, SIMULATED_SHIFT)
    ### ###

    ### Shift estimates with simulated images ###
    # transformed_image = ImageRun.simulated_generation("flowerfull.jpg", SIMULATED_SHIFT, RESIZE_VAR, GREY, WEIGHTING_SIM)
    ### ###
    # # Original image
    # original_image = cv2.imread("python/test_im/flowerfull.jpg")[:, :, 0]
    # psf = WeightingEstimate.get_img_psf(1-WEIGHTING_SIM, SIMULATED_SHIFT)
    # # Deconvolved image
    # deconvolved = WeightingEstimate.deconvolve_img(transformed_image[:, :, 0], psf)

    # # Perform 2D Fourier Transform on the original image
    # f_transform_original = np.fft.fft2(original_image)
    # f_transform_shifted_original = np.fft.fftshift(f_transform_original)
    # magnitude_spectrum_original = np.abs(f_transform_shifted_original)
    # log_magnitude_spectrum_original = np.log1p(magnitude_spectrum_original)

    # # Perform 2D Fourier Transform on the deconvolved image
    # f_transform_deconvolved = np.fft.fft2(deconvolved)
    # f_transform_shifted_deconvolved = np.fft.fftshift(f_transform_deconvolved)
    # magnitude_spectrum_deconvolved = np.abs(f_transform_shifted_deconvolved)
    # log_magnitude_spectrum_deconvolved = np.log1p(magnitude_spectrum_deconvolved)

    # # Plot each image separately

    # # Original Image
    # plt.figure(figsize=(6, 6))
    # plt.title('Original Image')
    # plt.imshow(original_image, cmap='gray')

    # # Fourier Magnitude Spectrum of Original Image
    # plt.figure(figsize=(6, 6))
    # plt.title('Fourier Magnitude Spectrum (Original)')
    # im1 = plt.imshow(log_magnitude_spectrum_original, norm=LogNorm(), cmap='plasma')
    # cbar1 = plt.colorbar(im1)
    # cbar1.set_label('Log Magnitude')

    # # Deconvolved Image
    # plt.figure(figsize=(6, 6))
    # plt.title('Deconvolved Image')
    # plt.imshow(deconvolved, cmap='gray')

    # # Fourier Magnitude Spectrum of Deconvolved Image
    # plt.figure(figsize=(6, 6))
    # plt.title('Fourier Magnitude Spectrum (Deconvolved)')
    # im2 = plt.imshow(log_magnitude_spectrum_deconvolved, norm=LogNorm(), cmap='plasma')
    # cbar2 = plt.colorbar(im2)
    # cbar2.set_label('Log Magnitude')
    # plt.show()
    ### Run estimation only getting w1 ###
    # ImageRun.run_estimate_w1(transformed_image)
    ### ###
    
    gt()
    rgb, r, g, b, w12 = PatchRun.process_all_chanels(transformed_image, PATCH_SIZE)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2,2,1)
    plt.imshow(r, cmap='gray')
    plt.title("Red Channel")
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.imshow(g, cmap='gray')
    plt.title("Green Channel")
    plt.axis('off')
    
    plt.subplot(2,2,3)
    plt.imshow(b, cmap='gray')
    plt.title("Blue Channel")
    plt.axis('off')
    
    plt.subplot(2,2,4)
    plt.imshow(rgb)
    plt.title("RGB Image")
    plt.axis('off')
    # _ = PatchGetAndCombine.create_full_quiver_with_overlap(rgb, transformed_image.shape[:2], (PATCH_SIZE, PATCH_SIZE), w12)
    plt.show()
    
    
if __name__ == "__main__":
    main()