import numpy as np
import cv2
import skimage as sk
import Autocorrelation as ac
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
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
        print("Normalising as max value has been reached in clarity both imgs")
        img = img / np.max(img)

    # deconvolved_image_w1 = sk.restoration.wiener(img, psf_w1, balance=0)
    # deconvolved_image_w2 = sk.restoration.wiener(img, psf_w2, balance=0)
    deconvolved_image_w1 = deconvolve_img(img, psf_w1)
    deconvolved_image_w2 = deconvolve_img(img, psf_w2)

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(deconvolved_image_w1, cmap='gray')
    # plt.title(f"Clarity: {clarity_loss(deconvolved_image_w1)}")
    # plt.subplot(1, 2, 2)
    # plt.imshow(deconvolved_image_w2, cmap='gray')
    # plt.title(f"Clarity: {clarity_loss(deconvolved_image_w2)}")

    clarity_w1 = clarity_loss(deconvolved_image_w1)
    clarity_w2 = clarity_loss(deconvolved_image_w2)

    ## return the weight depending on which clarity is lower
    print(f"Clarity w1: {clarity_w1} and Clarity w2: {clarity_w2}")
    print(f"Weight w1: {w1_est} and Weight w2: {1 - w1_est}")
    if clarity_w2 < clarity_w1:
        print("--------------------CLARITY CHANGED--------------------")
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
    loss = 1

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

def check_flatness(shift_vals, filtered_corr, shift_estimate):
    """
    The loss is also added if the curve is not flat around the peak.
    Also the standard deviation of the curve is added to the loss to be the flatness
    shift_vals: the shift values
    filtered_corr: the filtered correlation values
    shift_estimate: the shift estimate
    """

    pos_idx = np.where(shift_vals == shift_estimate)[0][0]
    neg_idx = np.where(shift_vals == -shift_estimate)[0][0]

    loss = 1
    # check the peak of shift vals
    loss += 20*(abs(filtered_corr[pos_idx]) + abs(filtered_corr[neg_idx]))
    # overall flatness from std
    CENTRAL_REG = 3
    central_region_idx = np.where((shift_vals > CENTRAL_REG) | (shift_vals < -CENTRAL_REG))[0]
    
    if isinstance(filtered_corr, list) or isinstance(filtered_corr, np.ndarray):
        try:
            non_central_vals = [filtered_corr[i] for i in central_region_idx if 0 <= i < len(filtered_corr)]
            if non_central_vals:
                flatness = 50 * np.std(non_central_vals)
                loss += flatness
            else:
                print("No valid non-central values found.")
        except IndexError as e:
            print(f"IndexError: {e}. Please check the central_region_idx and filtered_corr lists.")
        except TypeError as e:
            print(f"TypeError: {e}. Please ensure filtered_corr is an array or list and can be indexed.")
    else:
        print("filtered_corr is not an array or list, cannot index with central_region_idx.")
    
    return loss

def minimise_corr_vals(corr_vals, shift_vals):
    """
    Minimise the correlation by adding loss for all the correlation values where shift is not 0
    corr_vals: the correlation values
    shift_vals: the shift values
    """
    zero_idx = np.where(np.isclose(shift_vals, 0))[0].item()
    # for idx, shift in enumerate(shift_vals):
    #         if shift == 0:
    #             zero_idx = idx
    #             break
    loss = 1
    for i in range(0, zero_idx - 2):
        distance = zero_idx - i
        # if distance < 12*5:
        #     loss += distance * abs(corr_vals[i])
        # else:
        #     loss += abs(corr_vals[i])
        loss += distance * abs(corr_vals[i])
    for i in range(zero_idx + 2, len(corr_vals)):
        distance = i - zero_idx
        # if distance < 12*5:
        #     loss += distance * abs(corr_vals[i])
        # else:
        #     loss += abs(corr_vals[i])
        loss += distance * abs(corr_vals[i])
    return loss

def get_img_psf_2(w1, w2, shift):
    """
    Get the image psf. That can be applied to the image for deconvolution.
    w1: the weight of the psf
    shift: the shift value
    """
    psf = np.zeros(shift + 1)
    psf[-1] = w1
    psf[0] = w2

    # normalise
    psf = psf / np.sum(psf)

    psf = np.expand_dims(psf, axis=0)

    return psf

def loss_function_one_est(estimate, shifted_img, shift_val, loss_vals, w1_vals, all_losses):
    """
    Loss function to optimise the weights of the psf.
    Just for one w1 and the other is 1 - w1
    estimate: the estimate of the weight
    shifted_img: the shifted image
    shift_val: the shift value
    """
    if np.max(shifted_img) > 1:
        print("Normalising as max value has been reached in loss function one set")
        shifted_img = shifted_img / np.max(shifted_img)
    psf_estimate = get_img_psf(estimate, shift_val)
    # deconvolved_image = sk.restoration.wiener(shifted_img, psf_estimate, balance=0)
    # deconvolved_image = sk.restoration.richardson_lucy(shifted_img, psf_estimate, num_iter=1)
    deconvolved_image = deconvolve_img(shifted_img, psf_estimate)
    # print(f"Deconvolved image shape: {deconvolved_image.shape}")
    # normalise deconvolved image
    deconvolved_img_norm = deconvolved_image / np.max(deconvolved_image)
    # if np.isnan(deconvolved_img_norm).any():
    #     return 100
    # obtain tyhe correlation values of the deconvolved image
    shift_vals, corr_vals = ac.compute_auto_corr(deconvolved_img_norm, shift_val, shift_est_func=True, normalised=True)
    # Extract correlation values within shift values from -12 to 12
    # valid_indices = np.where((shift_vals >= -5) & (shift_vals <= 5))[0]
    
    # corr_vals = corr_vals[valid_indices]
    # shift_vals = shift_vals[valid_indices]
    
    # if np.max(corr_vals) - np.min(corr_vals) == 0:
    #     plt.figure()
    #     plt.imshow(shifted_img, cmap='gray')
    #     plt.figure()
    #     plt.imshow(deconvolved_img_norm, cmap='gray')
    #     plt.show()
    #     print(corr_vals)
    #     return 100
    
    corr_filt = ac.obtain_peak_highlighted_curve(corr_vals)
    # print(corr_vals.shape)
    # print(corr_filt.shape)
    # normalise corr_filt between 0 - 1
    # check if nan in corr_filt
    # range_corr_filt = np.max(corr_filt) - np.min(corr_filt)
    # if range_corr_filt != 0:
    #     corr_filt = (corr_filt - np.min(corr_filt)) / range_corr_filt
    # else:
    #     # If the values are identical, no need to normalize as they are already uniform
    #     corr_filt = corr_filt - np.min(corr_filt)

    # corr_filt = (corr_filt - np.min(corr_filt)) / range_corr_filt
    loss = 0.1*minimise_corr_vals(corr_vals, shift_vals)
    # loss += check_flatness(shift_vals, corr_filt, shift_val)
    # loss += check_gradients(corr_filt, shift_vals)
    
    pos_shift_idx = np.where(np.isclose(shift_vals, shift_val))[0].item()
    neg_shift_idx = np.where(np.isclose(shift_vals, -shift_val))[0].item()

    loss += 100*corr_filt[neg_shift_idx]
    loss += 100*corr_filt[pos_shift_idx]

    w1_vals.append(estimate)
    all_losses.append((estimate, loss))

    # min_val = np.where(loss_vals == np.min(loss_vals))[0][0]

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
    shift_val = shift_val
    all_losses = []
    
    result = sp.optimize.differential_evolution(loss_function_one_est, 
                                                bounds=[BOUNDS], 
                                                args=(shifted_img, shift_val, loss_vals, w1_vals, all_losses),
                                                disp=False,
                                                tol = 0.00001,
                                                # mutation=(0.9, 1.4),  # Higher mutation factor to explore more aggressively
                                                # recombination=0.4,
                                                polish=False, # use L-BFGS-B to polish the best result
                                                maxiter=75,
                                                popsize = 50,
                                                workers=30)
    # result = sp.optimize.minimize(
    #     loss_function_one_est, 
    #     x0=[0.5],  # Initial guess for w1
    #     args=(shifted_img, shift_val, loss_vals, w1_vals, all_losses),
    #     method='Powell',
    #     bounds=[BOUNDS],
    #     tol=0.0000001,
    #     options={'disp': True}
    # )
    
    est_w1 = result.x
    loss_value = result.fun
    
    # est_w1 = clarity_both_imgs(shifted_img, est_w1, shift_val)

    return est_w1, loss_value, all_losses

def deconvolve_img(img, psf, balance = 0):
    """
    Deconvolve the image with the psf
    img: the image to deconvolve
    psf: the psf to use
    balance: the balance parameter
    """
    # return sk.restoration.wiener(img, psf, balance=balance)
    return sk.restoration.richardson_lucy(img, psf, num_iter=1)