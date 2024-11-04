import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import uniform_filter
def apply_savgol_filter(corr_vals, window_size = 7, poly_order=3):
    """
    Apply a savgol filter to the correlation values. All it is doing is smoothing the curve
    corr_vals: the correlation values to apply the filter to
    window_size: the window size of the filter
    poly_order: the order of the polynomial to fit
    """
    
    return sp.signal.savgol_filter(corr_vals, window_size, poly_order)

def obtain_peak_highlighted_curve(corr_vals):
    """
    make the peaks of the curve more viisible taking away the smoothed parts
    corr_vals: the correlation values to apply the filter to
    """
    return np.abs(corr_vals - apply_savgol_filter(corr_vals))

def compute_auto_corr(img, est_shift_val, shift_est_func=False, normalised=True):
    """
    Compute the autocorrelation of an image
    img: image to compute the autocorrelation of
    est_shift_val: the estimated shift value if one hasnt been calculated yet shift_est_func=True
    shift_est_func: if True then we are estimating the shift value so need to compute the autocorrelation for the entire image
    shift_est__func: if False then just obtaining the cross correlation values for the deconvolved iamge
    normalised: if True then normalise the image before computing the autocorrelation
    """
    # if shift_est we are estimating the shfit so need to comput the autocorr for the entire image
    if shift_est_func:
        max_shift = img.shape[1]//2
    else:
        # times two because wanna get the extra parts around the peaks
        max_shift = est_shift_val*2
        # max_shift = img.shape[1]//2 # doing a test to see if larger max shift gives better results due to mroe "flatness"

    shift_values = []
    corr_values = []
    # loop through the shift values
    for x_shift in range(-max_shift, max_shift + 1):
        shifted_shifted_img = np.roll(img, x_shift, axis=1)     
        # shifted_shifted_img = sp.ndimage.shift(img, (0, x_shift), order=0, mode="constant") 
        if normalised:
            # perform np.correlate
            img_cent = img - np.mean(img)
            shifted_shifted_cent = shifted_shifted_img - np.mean(shifted_shifted_img)
            num = np.sum(img_cent*shifted_shifted_cent)
            den = np.sqrt(np.sum(img_cent**2)*np.sum(shifted_shifted_cent**2))
            cross_corr = num/den
            
            # mean1, stddev1 = cv2.meanStdDev(img)
            # mean2, stddev2 = cv2.meanStdDev(shifted_shifted_img)
            
            # norm_img1 = (img - mean1) / stddev1
            # norm_img2 = (shifted_shifted_img - mean2) / stddev2
            
            # cross_corr = np.sum(norm_img1 * norm_img2) / np.sqrt(np.sum(norm_img1**2) * np.sum(norm_img2**2))
            
        else:
            img_flat = img.flatten()
            shifted_shifted_flat = shifted_shifted_img.flatten()

            cross_corr = np.correlate(img_flat, shifted_shifted_flat, mode='valid')
            
        # add max value and shift value to list
        corr_values.append(np.max(cross_corr))
        shift_values.append((x_shift))
        
    return shift_values, corr_values
def compute_auto_corr(img, est_shift_val, shift_est_func=False, normalised=True, patch_size=5):
    """
    Compute the autocorrelation of an image using local means for normalization.
    
    img: image to compute the autocorrelation of
    est_shift_val: the estimated shift value if one hasn't been calculated yet, shift_est_func=True
    shift_est_func: if True, then we are estimating the shift value so need to compute the autocorrelation for the entire image
    normalised: if True, then normalize the image before computing the autocorrelation
    patch_size: Size of the sliding window (patch) for local mean calculation
    
    Returns:
    - shift_values: The shifts applied.
    - corr_values: The correlation values corresponding to each shift.
    """
    # If estimating shift, compute over the entire image; otherwise, use a specific shift range
    if shift_est_func:
        max_shift = img.shape[1] // 2
    else:
        max_shift = est_shift_val * 2

    shift_values = []
    corr_values = []

    # Helper function to calculate local mean and variance
    def local_mean_variance(image, patch_size):
        # Calculate local mean using uniform_filter for a sliding window
        local_mean = uniform_filter(image, size=patch_size)
        # Local variance (difference from mean squared) for normalization
        local_variance = uniform_filter(image**2, size=patch_size) - local_mean**2
        return local_mean, np.sqrt(np.maximum(local_variance, 1e-6))  # Variance can't be zero

    # Get the local mean and variance for the original image
    img_mean, img_std = local_mean_variance(img, patch_size)

    # Loop through the shift values
    for x_shift in range(-max_shift, max_shift + 1):
        # Shift the image horizontally
        shifted_img = np.roll(img, x_shift, axis=1)
        # shifted_img = sp.ndimage.shift(img, (0, x_shift), order=3, mode="constant") 
        if normalised:
            # Get local mean and variance for the shifted image
            shifted_mean, shifted_std = local_mean_variance(shifted_img, patch_size)

            # Center the images by subtracting their local means
            img_cent = img - img_mean
            shifted_cent = shifted_img - shifted_mean

            # Compute numerator (cross-correlation)
            num = np.sum(img_cent * shifted_cent)

            # Compute denominator (product of local standard deviations)
            # den = np.sum(img_std * shifted_std)
            den = np.sqrt(np.sum(img_cent**2) * np.sum(shifted_cent**2))

            # Avoid division by zero and calculate normalized cross-correlation
            cross_corr = num / den if den != 0 else 0
        else:
            # Without normalization, just compute the sum of products
            cross_corr = np.sum(img * shifted_img)

        shift_values.append(x_shift)
        corr_values.append(cross_corr)

    return shift_values, corr_values

