import Images
import ShiftEstimate
import WeightingEstimate
import Autocorrelation as ac
import matplotlib.pyplot as plt
import numpy as np
from skimage import restoration as sk

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

        image = transformed_image[:, :, i] / np.max(transformed_image[:, :, i])
        deconvolved = sk.wiener(image, WeightingEstimate.get_img_psf(w1_vals[i], shift_estimation), balance=0)

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

def run_estimate_w1_w2(transformed_image):
    """
    Run estimation getting w1 and w2
    transformed_image: the image to estimate the weighting for
    """

    shift_estimation = ShiftEstimate.compute_pixel_shift(transformed_image)

    w12_vals = []

    print(f"Shift estimate: {shift_estimation}")

    for i in range(3):
        img_channel = transformed_image[:, :, i]

        est = WeightingEstimate.optimise_psf_both_weight(img_channel, shift_estimation)

        w12_vals.append(est)

    for i in range(len(w12_vals)):
        print(f"Channel {i}: {w12_vals[i]}")
        plt.figure()
        plt.subplot(1, 3, 1)

        image = transformed_image[:, :, i] / np.max(transformed_image[:, :, i])
        deconvolved = sk.wiener(image, WeightingEstimate.get_img_psf_w1_w2(w12_vals[i][0], w12_vals[i][1], shift_estimation), balance=0)

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