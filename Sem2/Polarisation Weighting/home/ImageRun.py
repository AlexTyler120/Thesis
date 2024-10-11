import Images
import ShiftEstimate
import WeightingEstimate
import Autocorrelation as ac
import matplotlib.pyplot as plt
import numpy as np
from skimage import restoration as sk
import cv2
import pickle

def polarised_generation(file_name, degree, resize_var, grey, shift):
    """
    Generate a shifted image using two differently polarised images
    At least orthoganal to each other
    file_name: the file name of the first image
    degree: the degree of rotation of the second image
    resize_var: the amount to resize the image by
    grey: if True then convert the image to greyscale
    shift: the amount to shift the image by
    """
    # path1 = "python/test_im/" + file_name + "/rect_" + file_name + "_"+str(degree)+".png"
    # path2 = "python/test_im/" + file_name + "/rect_" + file_name + "_"+str(degree+45)+".png"
    # path3 = "python/test_im/" + file_name + "/rect_" + file_name + "_"+str(degree+90)+".png"
    # path4 = "python/test_im/" + file_name + "/rect_" + file_name + "_"+str(degree+135)+".png"
    
    # i0 = cv2.imread(path1)
    # i45 = cv2.imread(path2)
    # i90 = cv2.imread(path3)
    # i135 = cv2.imread(path4)
    # # convert to rgb
    # I0 = cv2.cvtColor(i0, cv2.COLOR_BGR2RGB)
    # I45 = cv2.cvtColor(i45, cv2.COLOR_BGR2RGB)
    # I90 = cv2.cvtColor(i90, cv2.COLOR_BGR2RGB)
    # I135 = cv2.cvtColor(i135, cv2.COLOR_BGR2RGB)
    
    # I90[:, shift:] = I90[:, :-shift]
    # I135[shift:, :] = I135[:-shift, :]
    # I45[:-shift, :] = I45[shift:, :]
    # I0[:, :-shift] = I0[:, shift:]
    
    # I0I90 = cv2.addWeighted(I0, 0.5, I90, 0.5, 0)
    # I45I135 = cv2.addWeighted(I45, 0.5, I135, 0.5, 0)
    # total = cv2.addWeighted(I0I90, 0.5, I45I135, 0.5, 0)
    # plt.figure()
    # plt.imshow(total)
    # plt.title("Four Polarised Images Combined")
    # plt.show()

    path1 = "python/test_im/" + file_name + "/rect_" + file_name + "_"+str(degree)+".png"
    path2 = "python/test_im/" + file_name + "/rect_" + file_name + "_"+str(degree+90)+".png"
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
    resize_var = 1

    # make image larger
    img_channel_grey = cv2.resize(img_channel_grey, (img_channel_grey.shape[1] * resize_var, img_channel_grey.shape[0] * resize_var))
    shift_estimation = shift_estimation * resize_var
    
    # shift, corr = ac.compute_auto_corr(img_channel_grey, shift_estimation, shift_est_func=True, normalised=True)
    # valid_index = np.where((shift >= -5) & (shift <= 5))[0]
    # corr_filt = ac.obtain_peak_highlighted_curve(corr)
    
    # shift = shift[valid_index]
    # corr = corr[valid_index]
    # corr_filt = corr_filt[valid_index]
    # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # im = axs[1, 0].imshow(img_channel_grey[5:-5, 5:-5], cmap='gray')
    # axs[1, 0].set_title(f"Channel {channel} Patch Blurred")
    # # plt.colorbar(im, ax=axs[0, 0])
    
    # axs[1, 1].plot(shift, corr)
    # axs[1, 1].set_title(f"Channel {channel} Patch Cross-Correlation")
    
    # axs[1, 2].plot(shift, corr_filt)
    # axs[1, 2].set_title(f"Channel {channel} Patch Filtered Correlation")
    
    est1, loss, _ = WeightingEstimate.optimise_psf(img_channel_grey, shift_estimation)
    est2 = 1 - est1
    w12_vals = [est1, est2]
    print(f"psf {WeightingEstimate.get_img_psf(est1, shift_estimation)}")
    deconvolved = WeightingEstimate.deconvolve_img(img_channel_grey, WeightingEstimate.get_img_psf(est1, shift_estimation))
    
    deconvolved = cv2.resize(deconvolved, (patch.shape[1], patch.shape[0]))
    shift_estimation = shift_estimation // resize_var
    
    # im = axs[0, 0].imshow(og_patch[5:-5, 5:-5], cmap='gray')
    # axs[0, 0].set_title(f"Channel {channel} Patch Original")
    # # plt.colorbar(im, ax=axs[1, 0])
    
    # shift, corr = ac.compute_auto_corr(og_patch, shift_estimation, shift_est_func=True, normalised=True)
    # shift = shift[valid_index]
    # corr = corr[valid_index]
    # axs[0, 1].plot(shift, corr)
    # axs[0, 1].set_title(f"Channel {channel} Patch Original Cross-Correlation")
    
    # corr_filt = ac.obtain_peak_highlighted_curve(corr)
    # # corr_filt = corr_filt[valid_index]
    # axs[0, 2].plot(shift, corr_filt)
    # axs[0, 2].set_title(f"Channel {channel} Patch Original Filtered Correlation")
    
    # plt.tight_layout()
    
    # plt.show()
    
    
    # im = axs[1, 0].imshow(deconvolved[5:-5, 5:-5], cmap='gray')
    # axs[1, 0].set_title(f"Channel {channel} Patch Deconvolved")
    # plt.colorbar(im, ax=axs[1, 0])
    
    # shift, corr = ac.compute_auto_corr(deconvolved, shift_estimation, shift_est_func=True, normalised=True)
    # shift = shift[valid_index]
    # corr = corr[valid_index]
    # axs[1, 1].plot(shift, corr)
    # axs[1, 1].set_title(f"Channel {channel} Patch Deconvolved Cross-Correlation")
    
    # corr_filt = ac.obtain_peak_highlighted_curve(corr)
    # # corr_filt = corr_filt[valid_index]
    # axs[1, 2].plot(shift, corr_filt)
    # axs[1, 2].set_title(f"Channel {channel} Patch Deconvolved Filtered Correlation")
    
    # plt.tight_layout()
    
    # plt.show()
    
    # # stack the deconvolved images
    return deconvolved, w12_vals

