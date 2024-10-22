import Images
import ShiftEstimate
import WeightingEstimate
import Autocorrelation as ac
import matplotlib.pyplot as plt
import numpy as np
from skimage import restoration as sk
import cv2
import pickle
import scipy as sp

def polarised_generation(file_name, prefix, degree, resize_var, grey, shift):
    """
    Generate a shifted image using two differently polarised images
    At least orthoganal to each other
    file_name: the file name of the first image
    degree: the degree of rotation of the second image
    resize_var: the amount to resize the image by
    grey: if True then convert the image to greyscale
    shift: the amount to shift the image by
    """
    path1 = "python/test_im/" + file_name +"/"+ prefix + file_name + "_"+str(degree)+".png"
    path2 = "python/test_im/" + file_name+"/" + prefix + file_name + "_"+str(degree+90)+".png"
    print(path1)
    print(path2)
    img1, _, _ = Images.read_image(path1, resize_var, grey)
    img2, _, _ = Images.read_image(path2, resize_var, grey)

    transformed_image = Images.create_shifted_image_polarised_imgs(img1, img2, shift, True)
    # transformed_image = Images.create_shifted_image_polarised_y(img1, img2, shift, True)

    return transformed_image

def simulated_generation(file_name, shift, resize_var, grey, weighting):
    """
    Generate a shifted image using a single image
    file_name: the file name of the image
    shift: the amount to shift the image by
    resize_var: the amount to resize the image by
    grey: if True then convert the image to greyscale
    weighting: the weighting to apply to the shift
    """

    path = "python/test_im/" + file_name

    img, _, _ = Images.read_image(path, resize_var, grey)

    transformed_image = Images.create_shifted_simulation(img, weighting, shift)

    return transformed_image


def run_estimate_w1(transformed_image):
    """
    Run estimation only getting w1
    transformed_image: the image to estimate the weighting for
    """

    shift_estimation = ShiftEstimate.compute_pixel_shift(transformed_image)
    w1_vals = []

    print(f"Shift estimate: {shift_estimation}")

    for i in range(3):
        img_channel = transformed_image[:, :, i]

        est, _, all_losses = WeightingEstimate.optimise_psf(img_channel, shift_estimation)
        
        with open(f'loss_values_{i}_powell_03.pkl', 'wb') as f:
            pickle.dump(all_losses, f)
            
        w1_vals.append(est)

    for i in range(len(w1_vals)):
        print(f"Channel {i}: {w1_vals[i]}")
        plt.figure()
        plt.subplot(1, 3, 1)

        deconvolved = sk.wiener(transformed_image[:, :, i], WeightingEstimate.get_img_psf(w1_vals[i], shift_estimation), balance=0)

        plt.imshow(deconvolved, cmap='gray')
        plt.title(f"Channel {i} Deconvolved")
        
        plt.subplot(1, 3, 2)
        shift, corr = ac.compute_auto_corr(deconvolved, shift_estimation)
        plt.plot(shift, corr)
        plt.title(f"Channel {i} Auto Correlation")
        
        plt.subplot(1, 3, 3)
        plt.plot(shift, ac.obtain_peak_highlighted_curve(corr))
        plt.title(f"Channel {i} Filtered Auto Correlation")

    plt.show()
    
def run_estimate_w1_w2_patch(patch, channel, shift_estimation, og_patch = None):
    """
    Run estimation getting w1 and w2
    transformed_image: the image to estimate the weighting for
    """

    w12_vals = []

    print(f"Shift estimate: {shift_estimation}")
    
    img_channel_grey = patch[:, :, channel]
    # save the original patch
    # plt.imsave("patch.png", img_channel_grey, cmap='gray')
    resize_var = 1
    
    # normalise og_patch
    if og_patch is not None:
        og_patch = og_patch[:,:, channel]
        # og_patch = cv2.normalize(og_patch, None, 0, 1, cv2.NORM_MINMAX)
        # og_patch = og_patch / np.max(og_patch)
    # make image larger
    img_channel_grey = cv2.resize(img_channel_grey, (img_channel_grey.shape[1] * resize_var, img_channel_grey.shape[0] * resize_var))
    # histogram equalisation
    shift_estimation = shift_estimation * resize_var
    
    shift, corr = ac.compute_auto_corr(img_channel_grey, shift_estimation, shift_est_func=True, normalised=True)
    shift = np.array(shift)
    corr = np.array(corr)

    valid_indicies = []
    for i in range(len(shift)):
        if np.abs(shift[i]) <= 6:
            valid_indicies.append(i)
    valid_indicies = np.array(valid_indicies)
    
    if og_patch is not None:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        # Increase contrast using histogram equalization
        # img_channel_grey = cv2.equalizeHist((img_channel_grey * 255).astype(np.uint8)) / 255.0
        im = axs[0, 0].imshow(img_channel_grey[6:-6, 6:-6], cmap='gray', vmin=0, vmax=1)
        axs[0, 0].set_title(f"Channel {channel} Patch Blurred")
        # plt.colorbar(im, ax=axs[0])
        axs[1, 1].plot(shift[valid_indicies], corr[valid_indicies], label = "Blurred")
        axs[1, 1].set_title(f"Channel {channel} Patch Cross-Correlation")
    
    # est1, loss, _ = WeightingEstimate.optimise_psf(img_channel_grey, shift_estimation)
    # est2 = 1 - est1
    # w12_vals = [est1, est2]
    
    est, loss, _ = WeightingEstimate.optimise_psf_2(img_channel_grey, shift_estimation)
    w12_vals = [est[0], est[1]]
    # w12_vals = [0.5, 0.4]
    # deconvolved = WeightingEstimate.deconvolve_img(img_channel_grey, WeightingEstimate.get_img_psf(est1, shift_estimation))
    

    deconvolved = WeightingEstimate.deconvolve_img(img_channel_grey, WeightingEstimate.get_img_psf_2(w12_vals[0], w12_vals[1], shift_estimation))
    
    # deconvolved = cv2.resize(deconvolved, (patch.shape[1], patch.shape[0]))
    # shift_estimation = shift_estimation // resize_var
    
    if og_patch is not None:
        shift_dec, corr_dec = ac.compute_auto_corr(deconvolved, shift_estimation, shift_est_func=True, normalised=True)
        shift_dec = np.array(shift_dec)
        corr_dec = np.array(corr_dec)
        corr_diffs = []
        im = axs[0, 1].imshow(og_patch[6:-6, 6:-6], cmap='gray', vmin=0, vmax=1)
        axs[0, 1].set_title(f"Channel {channel} Patch Original")
        # plt.colorbar(im, ax=axs[0, 1])
        shift_og, corr_og = ac.compute_auto_corr(og_patch, shift_estimation, shift_est_func=True, normalised=True)
        shift_og = np.array(shift_og)
        corr_og = np.array(corr_og)
        axs[1, 1].plot(shift_og[valid_indicies], corr_og[valid_indicies], label="Original")
        axs[1, 1].set_title(f"Channel {channel} Patch Original Cross-Correlation")
        # Increase contrast using histogram equalization for the deconvolved image
        im = axs[1, 0].imshow(deconvolved[6:-6, 6:-6], cmap='gray', vmin=0, vmax=1)
        axs[1, 0].set_title(f"Channel {channel} Patch Deconvolved")
        # plt.colorbar(im, ax=axs[1, 0])
        axs[1, 1].plot(shift_dec[valid_indicies], corr_dec[valid_indicies], label="Deconvolved")
        axs[1,1].legend()
        # # stack the deconvolved images
        plt.tight_layout()
        plt.figure()
        convolved_blur = sp.signal.fftconvolve(deconvolved, WeightingEstimate.get_img_psf_2(w12_vals[0], w12_vals[1], shift_estimation), mode='same')
        plt.imshow(convolved_blur, cmap='gray', vmin=0, vmax=1)
        plt.title("Deconvolved Image Convolved with PSF")
        plt.show()
    
    # dec_other = WeightingEstimate.deconvolve_img(img_channel_grey, WeightingEstimate.get_img_psf(est2, shift_estimation), wiener=True)
    print(w12_vals)
    return deconvolved, w12_vals, None

def run_estimate_w1_w2_colour(patch, shift_estimation, og_patch = None):
    w12_vals = []

    print(f"Shift estimate: {shift_estimation}")

    # shift, corr = ac.compute_auto_corr(patch[6:-6, 6:-6], shift_estimation, shift_est_func=True, normalised=True)
    # corr_filt = ac.obtain_peak_highlighted_curve(corr)
    # fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    # im = axs[0, 0].imshow(patch[6:-6, 6:-6])
    # axs[0, 0].set_title(f"Patch Blurred")
    # axs[1, 1].plot(shift, corr, label = "Blurred")
    # axs[1, 1].set_title(f"Patch Cross-Correlation")
    
    
    est1, loss, _ = WeightingEstimate.optimise_psf(patch, shift_estimation)
    est2 = 1 - est1
    w12_vals = [est1, est2]
    print(f"psf {WeightingEstimate.get_img_psf(est1, shift_estimation)}")
    deconvolved = WeightingEstimate.deconvolve_img_colour(patch, WeightingEstimate.get_img_psf(est1, shift_estimation))
    
    
    # im = axs[0, 1].imshow(og_patch[6:-6, 6:-6])
    # axs[0, 1].set_title(f"Patch Original")    
    # shift, corr = ac.compute_auto_corr(og_patch[6:-6, 6:-6], shift_estimation, shift_est_func=True, normalised=True)
    # axs[1, 1].plot(shift, corr, label="Original")
    # axs[1, 1].set_title(f"Patch Original Cross-Correlation")
    # im = axs[1, 0].imshow(deconvolved[6:-6, 6:-6])
    # axs[1, 0].set_title(f"Patch Deconvolved")    
    # shift, corr = ac.compute_auto_corr(deconvolved[6:-6, 6:-6], shift_estimation, shift_est_func=True, normalised=True)
    # axs[1, 1].plot(shift, corr, label="Deconvolved")
    # axs[1,1].legend()
    
    
    # plt.tight_layout()
    
    # plt.show()
    return deconvolved, w12_vals