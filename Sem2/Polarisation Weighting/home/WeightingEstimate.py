import numpy as np
import cv2
import skimage as sk
import Autocorrelation as ac
import scipy as sp
import matplotlib.pyplot as plt

def get_img_psf(w1, shift):
    """
    Get the image psf. That can be applied to the image for deconvolution.
    w1: the weight of the psf
    shift: the shift value
    """
    psf = np.zeros(shift + 1)
    psf[-1] = w1
    psf[0] = 1 - w1

    # normalise
    psf = psf / np.sum(psf)

    psf = np.expand_dims(psf, axis=0)

    return psf

def clarity_loss(img):
    """
    Compute the clarity loss of the image. This is done by using sobel of the image.
    A higher sobel means a less clear image
    img: the deconvolved image to compute the clarity loss of
    """
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    return np.mean(sobel)

def clarity_both_imgs(img, w1_est, shift_val):
    """
    Compute the clarity loss of image with w1 and 1-w1
    img: the deconvolved image
    w1_est: the estimate of the weight
    shift_val: the shift value
    """
    psf_w1 = get_img_psf(w1_est, shift_val)
    psf_w2 = get_img_psf(1 - w1_est, shift_val)

    if np.max(img) > 1:
        img = img / np.max(img)

    deconvolved_image_w1 = sk.restoration.wiener(img, psf_w1, balance=0)
    deconvolved_image_w2 = sk.restoration.wiener(img, psf_w2, balance=0)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(deconvolved_image_w1, cmap='gray')
    plt.title(f"Clarity: {clarity_loss(deconvolved_image_w1)}")
    plt.subplot(1, 2, 2)
    plt.imshow(deconvolved_image_w2, cmap='gray')
    plt.title(f"Clarity: {clarity_loss(deconvolved_image_w2)}")

    clarity_w1 = clarity_loss(deconvolved_image_w1)
    clarity_w2 = clarity_loss(deconvolved_image_w2)

    ## return the weight depending on which clarity is lower
    print(f"Clarity w1: {clarity_w1} and Clarity w2: {clarity_w2}")
    print(f"Weight w1: {w1_est} and Weight w2: {1 - w1_est}")
    if clarity_w2 > clarity_w1:
        return 1 - w1_est
    else:
        return w1_est


def check_gradients(corr_vals, shift_vals):
    """
    Check the gradients of the correlation values
    corr_vals: the correlation values
    shift_vals: the shift values
    """
    zero_idx = np.where(shift_vals == 0)[0][0]
    loss = 0

    # check the gradients of the correlation values
    # if the gradients are not correct then add to the loss
    # the gradients should be increasing to the left of the peak and decreasing to the right
    for i in range(1, zero_idx):
        if corr_vals[i] <= corr_vals[i-1]:
            loss += abs(corr_vals[i] - corr_vals[i-1])

    for i in range(zero_idx + 1, len(corr_vals)):
        if corr_vals[i] >= corr_vals[i-1]:
            loss += abs(corr_vals[i] - corr_vals[i-1])
    return loss

def check_flatness(shift_vals, fitlered_corr, shift_estimate):
    """
    The loss is also added if the curve is not flat around the peak.
    Also the standard deviation of the curve is added to the loss to be the flatness
    shift_vals: the shift values
    fitlered_corr: the filtered correlation values
    shift_estimate: the shift estimate
    """
    pos_idx = np.where(shift_vals == shift_estimate)[0][0]
    neg_idx = np.where(shift_vals == -shift_estimate)[0][0]

    loss = 0
    # check the peak of shift vals
    loss += 2*(abs(fitlered_corr[pos_idx]) + abs(fitlered_corr[neg_idx]))

    # overall flatness from std
    CENTRAL_REG = 3
    central_region_idx = np.where((shift_vals > CENTRAL_REG) & (-shift_vals < -CENTRAL_REG))[0]
    non_central_vals = fitlered_corr[central_region_idx]
    flatness = 50 * np.std(non_central_vals)
    loss += flatness
    return loss

def loss_function_one_est(estimate, shifted_img, shift_val, loss_vals, w1_vals):
    """
    Loss function to optimise the weights of the psf.
    Just for one w1 and the other is 1 - w1
    estimate: the estimate of the weight
    shifted_img: the shifted image
    shift_val: the shift value
    """
    if np.max(shifted_img) > 1:
        shifted_img = shifted_img / np.max(shifted_img)
    
    psf_estimate = get_img_psf(estimate, shift_val)

    deconvolved_image = sk.restoration.wiener(shifted_img, psf_estimate, balance=0)

    # normalise deconvolved image
    deconvolved_img_norm = deconvolved_image / np.max(deconvolved_image)

    # obtain tyhe correlation values of the deconvolved image
    shift_vals, corr_vals = ac.compute_auto_corr(deconvolved_img_norm, shift_val, shift_est_func=False, normalised=True)

    # apply a savgol filter to the correlation values for peak detection
    corr_filt = ac.obtain_peak_highlighted_curve(corr_vals)

    # develop loss
    loss = 0
    loss = check_gradients(corr_filt, shift_vals)
    loss += check_flatness(shift_vals, corr_filt, shift_val)

    loss_vals.append(loss)
    w1_vals.append(estimate)

    min_val = np.where(loss_vals == np.min(loss_vals))[0][0]

    # print(f"Loss: {loss.min()} and w1: {w1_vals[min_val]}")

    return loss

def optimise_psf(shifted_img, shift_val):
    """
    Optimising the w1 value for the PSF
    shifted_img: the shifted image
    shift_val: the shift value
    """

    BOUNDS = (0, 1)
    loss_vals = []
    w1_vals = []
    result = sp.optimize.differential_evolution(loss_function_one_est, 
                                                bounds=[BOUNDS], 
                                                args=(shifted_img, shift_val, loss_vals, w1_vals),
                                                disp=True,
                                                polish=True, # use L-BFGS-B to polish the best result
                                                workers=-1)
    
    w1_estimate = result.x
    loss_value = result.fun
    
    est_w1 = clarity_both_imgs(shifted_img, w1_estimate, shift_val)

    return est_w1

