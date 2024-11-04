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
    psf[0] = w1
    psf[-1] = 1 - w1

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
    for i in range(0, zero_idx - 1):
        distance = zero_idx - i
        loss += distance * abs(corr_vals[i])
    for i in range(zero_idx + 1, len(corr_vals)):
        distance = i - zero_idx
        loss += distance * abs(corr_vals[i])
    return loss


def loss_function_one_est(estimate, shifted_img, shift_val, loss_vals, w1_vals, neighweight):
    """
    Loss function to optimise the weights of the psf.
    Just for one w1 and the other is 1 - w1
    estimate: the estimate of the weight
    shifted_img: the shifted image
    shift_val: the shift value
    """
    # check num channels of shifted_img
    
    crop_val = 6
        
    psf_estimate = get_img_psf(estimate, shift_val)
    if len(shifted_img.shape) == 3:
        deconvolved_image = deconvolve_img_colour(shifted_img, psf_estimate)
    else:
        deconvolved_image = deconvolve_img(shifted_img, psf_estimate)

    # deconvolved_image = deconvolved_image / np.max(deconvolved_image)

    shift_vals, corr_vals = ac.compute_auto_corr(deconvolved_image, shift_val, shift_est_func=True, normalised=True)
    
    # valid_index = np.where((np.abs(shift_vals) <= 6) & (shift_vals != 0))[0]
    # valid_index = valid_index.astype(int)
    # print(valid_index)

    
    
    reg = minimise_corr_vals(corr_vals, shift_vals)
    reg2 = corr_vals[shift_vals == 5] + corr_vals[shift_vals == -5]
    cropped_dec = deconvolved_image[6:-6, 6:-6]
    convolved = sp.signal.fftconvolve(deconvolved_image, psf_estimate, mode = 'same')
    cropped_conv = convolved[6:-6, 6:-6]
    
    
    reg3 = 0 
    # if neighweight:
    #     avg_w1, avg_w2 = neighweight
    #     reg3 = 1 * ((avg_w1 - estimate)**2 + (avg_w2 - (1 - estimate))**2)
    loss = np.abs((shifted_img[crop_val:-crop_val] - convolved[crop_val:-crop_val]).sum()) + 10*reg + reg3
    # loss = 10*reg + np.abs(corr_vals[shift_vals == 10]) + np.abs(corr_vals[shift_vals == -10])
    # loss = reg

    w1_vals.append(estimate.item())
    loss_vals.append(loss)

    return loss


def optimise_psf(shifted_img, shift_val, neighbor_weights = None):
    """
    Optimising the w1 value for the PSF
    shifted_img: the shifted image
    shift_val: the shift value
    """

    BOUNDS = (0, 1)
    loss_vals = []
    w1_vals = []
    shift_val = shift_val
    
    
    result = sp.optimize.differential_evolution(loss_function_one_est, 
                                                bounds=[BOUNDS], 
                                                args=(shifted_img, shift_val, loss_vals, w1_vals, neighbor_weights),
                                                disp=False,
                                                tol = 0.000001,
                                                mutation=(0.9, 1.4),  # Higher mutation factor to explore more aggressively
                                                # recombination=0.4,
                                                polish=False, # use L-BFGS-B to polish the best result
                                                maxiter=50,
                                                popsize = 15,
                                                workers=4)
    # result = sp.optimize.minimize(
    #     loss_function_one_est, 
    #     x0=[0.5],  # Initial guess for w1
    #     args=(shifted_img, shift_val, loss_vals, w1_vals, neighbor_weights),
    #     method='Powell',
    #     bounds=[BOUNDS],
    #     tol=0.00000001,
    #     options={'disp': True}
    # )
    
    est_w1 = result.x
    loss_value = result.fun
    
    # est_w1 = clarity_both_imgs(shifted_img, est_w1, shift_val)

    return est_w1, loss_value, loss_vals, w1_vals

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

def loss_function_two_est(estimate, shifted_img, shift_val, loss_vals, w1_vals, all_losses, balance):
    """
    Loss function to optimise the weights of the psf.
    Just for one w1 and the other is 1 - w1
    estimate: the estimate of the weight
    shifted_img: the shifted image
    shift_val: the shift value
    """
    crop_val = 6
    # check num channels of shifted_img
    w1, w2 = estimate
    
    psf_estimate = get_img_psf_2(w1, w2, shift_val)
    
    if len(shifted_img.shape) == 3:
        deconvolved_image = deconvolve_img_colour(shifted_img, psf_estimate)
    else:
        deconvolved_image = deconvolve_img(shifted_img, psf_estimate, ri_iter=balance)

    # deconvolved_image = deconvolved_image / np.max(deconvolved_image)

    shift_vals, corr_vals = ac.compute_auto_corr(deconvolved_image, shift_val, shift_est_func=True, normalised=True)
    
    valid_index = np.where((np.abs(shift_vals) <= crop_val) & (shift_vals != 0))[0]
    valid_index = valid_index.astype(int)
    # print(valid_index)
    shift_vals_corrected = []
    corr_vals_corrected = []
    
    for index in valid_index:
        shift_vals_corrected.append(shift_vals[index])
        corr_vals_corrected.append(corr_vals[index])
    
    reg = minimise_corr_vals(corr_vals_corrected, shift_vals_corrected)
    # pos_shift_idx = np.where(np.isclose(shift_vals, shift_val))[0].item()
    # neg_shift_idx = np.where(np.isclose(shift_vals, -shift_val))[0].item()
    
    cropped_dec = deconvolved_image[crop_val:-crop_val, crop_val:-crop_val]
    convolved = sp.signal.fftconvolve(deconvolved_image, psf_estimate, mode='same')
    cropped_conv = convolved[crop_val:-crop_val, crop_val:-crop_val]

    # if (w1 > 0.6 and w2 < 0.4) or (w1 < 0.4 and w2 > 0.6):
    #     reg += 0.1
    # else:
    #     reg += 100
    # loss = (((shifted_img[crop_val:-crop_val, crop_val:-crop_val] - cropped_conv).sum())**2) + 10*reg
    loss = ((shifted_img[crop_val:-crop_val] - convolved[crop_val:-crop_val]).sum()) + 10*reg
    # loss = reg0

    # loss += np.abs(corr_vals[neg_shift_idx])
    # loss += np.abs(corr_vals[pos_shift_idx])

    w1_vals.append(estimate)
    all_losses.append((estimate, loss))

    return loss

def optimise_psf_2(shifted_img, shift_val, balance = 2):
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
    
    # result = sp.optimize.differential_evolution(loss_function_two_est, 
    #                                             bounds=[BOUNDS, BOUNDS], 
    #                                             args=(shifted_img, shift_val, loss_vals, w1_vals, all_losses, balance),
    #                                             disp=False,
    #                                             tol = 0.000001,
    #                                             # mutation=(0.9, 1.4),  # Higher mutation factor to explore more aggressively
    #                                             # recombination=0.4,
    #                                             polish=False, # use L-BFGS-B to polish the best result
    #                                             maxiter=50,
    #                                             popsize = 8,
    #                                             workers=4)
    
    result = sp.optimize.minimize(
        loss_function_two_est, 
        x0=[0.5, 0],  # Initial guess for w1 w2
        args=(shifted_img, shift_val, loss_vals, w1_vals, all_losses, balance),
        method='Powell',
        bounds=[BOUNDS, BOUNDS],
        tol=0.000001,
        options={'disp': True}
    )
    
    est = result.x
    # est_w1 = est[0]
    # est_w2 = est[1]
    loss_value = result.fun
    
    # est_w1 = clarity_both_imgs(shifted_img, est_w1, shift_val)

    return est, loss_value, all_losses
     

def deconvolve_img(img, psf, balance = 0.01, wiener = False, ri_iter = 2):
    """
    Deconvolve the image with the psf
    img: the image to deconvolve
    psf: the psf to use
    balance: the balance parameter
    """
    return sk.restoration.wiener(img, psf, balance=balance)
    if wiener:
        return sk.restoration.wiener(img, psf, balance=balance)
    else:
        return sk.restoration.richardson_lucy(img, psf, num_iter=ri_iter)
    
def deconvolve_img_colour(img, psf, balance = 0):
    """
    Deconvolve the image with the psf
    img: the image to deconvolve
    psf: the psf to use
    balance: the balance parameter
    """
    deconvolved_img = np.zeros(img.shape)
    for i in range(3):
        deconvolved_img[:, :, i] = deconvolve_img(img[:, :, i], psf, balance)
    return deconvolved_img