import Images
import ShiftEstimate
import WeightingEstimate
import Autocorrelation as ac
import matplotlib.pyplot as plt
import numpy as np
from skimage import restoration as sk
import cv2

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

    path1 = "python/test_im/" + file_name + "/" + file_name + "_"+str(degree)+".png"
    path2 = "python/test_im/" + file_name + "/" + file_name + "_"+str(degree+90)+".png"

    img1, _, _ = Images.read_image(path1, resize_var, grey)
    img2, _, _ = Images.read_image(path2, resize_var, grey)

    transformed_image = Images.create_shifted_image_polarised_imgs(img1, img2, shift, True)

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
    shift_estimation = 8
    w1_vals = []

    print(f"Shift estimate: {shift_estimation}")

    for i in range(3):
        img_channel = transformed_image[:, :, i]

        est = WeightingEstimate.optimise_psf(img_channel, shift_estimation)

        w1_vals.append(est)

    for i in range(len(w1_vals)):
        print(f"Channel {i}: {w1_vals[i]}")
        plt.figure()
        plt.subplot(1, 3, 1)

        # image = transformed_image[:, :, i] / np.max(transformed_image[:, :, i])
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
    
def run_estimate_w1_w2_patch(patch, channel, shift_estimation):
    """
    Run estimation getting w1 and w2
    transformed_image: the image to estimate the weighting for
    """

    w12_vals = []

    print(f"Shift estimate: {shift_estimation}")
    losses = []
    
    img_channel_grey = patch[:, :, channel]
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    # img_clahe = (img_channel_grey* 65535).astype(np.uint16)
    # img_channel = clahe.apply(img_clahe)
    # # Normalize the image to the range [0, 1]
    # img_channel = img_channel / np.max(img_channel)

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(img_channel, cmap='gray')
    

    est1, est2, loss = WeightingEstimate.optimise_psf_both_weight(img_channel_grey, shift_estimation)
    w12_vals.append([est1, est2])
    losses.append(loss)

    # deconvolve
    deconvolved = sk.wiener(img_channel_grey, WeightingEstimate.get_img_psf_w1_w2(est1, est2, shift_estimation), balance=0)
    # plt.subplot(1, 2, 2)
    # plt.imshow(deconvolved, cmap='gray')
    # plt.show()
    # stack the deconvolved images
    return deconvolved, w12_vals

def run_estimate_w1_w2(transformed_image):
    """
    Run estimation getting w1 and w2
    transformed_image: the image to estimate the weighting for
    """

    shift_estimation = ShiftEstimate.compute_pixel_shift(transformed_image)
    # shift_estimation = 5
    w12_vals = []
    deconvolved_all = []
    print(f"Shift estimate: {shift_estimation}")
    losses = []
    
    for i in range(3):
        img_channel_grey = transformed_image[:, :, i]
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
        # img_clahe = (img_channel_grey* 65535).astype(np.uint16)
        # img_channel = clahe.apply(img_clahe)
        # Normalize the image to the range [0, 1]
        img_channel = img_channel_grey / np.max(img_channel_grey)

        # plt.figure()
        # plt.imshow(img_channel)
        # plt.show()

        est1, est2, loss = WeightingEstimate.optimise_psf_both_weight(img_channel, shift_estimation)
        w12_vals.append([est1, est2])
        losses.append(loss)

        # deconvolve
        # deconvolved = sk.wiener(img_channel, WeightingEstimate.get_img_psf_w1_w2(est1, est2, shift_estimation), balance=0)
        # deconvolved_all.append(deconvolved)
    # stack the deconvolved images
    print(f"Channel 0 Loss: {losses[0]}\nChannel 1 Loss: {losses[1]}\nChannel 2 Loss: {losses[2]}")
    for i in range(len(w12_vals)):
        print(f"Channel {i}: {w12_vals[i]}")
        plt.figure()
        plt.subplot(1, 3, 1)

        # image = transformed_image[:, :, i] / np.max(transformed_image[:, :, i])
        deconvolved = sk.wiener(transformed_image[:, :, i], WeightingEstimate.get_img_psf_w1_w2(w12_vals[i][0], w12_vals[i][1], shift_estimation), balance=0)
        deconvolved_all.append(deconvolved)
        plt.imshow(deconvolved, cmap='gray')
        plt.title(f"Channel {i} Deconvolved")
        
        plt.subplot(1, 3, 2)
        shift, corr = ac.compute_auto_corr(deconvolved, shift_estimation)
        plt.plot(shift, corr)
        plt.title(f"Channel {i} Auto Correlation")
        
        plt.subplot(1, 3, 3)
        plt.plot(shift, ac.obtain_peak_highlighted_curve(corr))
        plt.title(f"Channel {i} Filtered Auto Correlation")

    # show the combined deconvolved image
    # plt.figure()
    # proper_final = np.dstack(deconvolved_all) * 255
    # proper_final = np.clip(proper_final, 0, 255).astype(np.uint8)

    # plt.imshow(cv2.cvtColor(proper_final, cv2.COLOR_BGR2RGB))
    # plt.title("Combined Deconvolved Image")
    plt.show()

    return np.dstack(deconvolved_all)
