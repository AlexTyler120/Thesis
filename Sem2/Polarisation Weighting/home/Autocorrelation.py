import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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
        # shift the iamge
        # shifted_shifted_img = sp.ndimage.shift(img, shift=(0,x_shift), mode='constant', cval=0)
        # if x_shift > 0:
        #     shifted_shifted_img = np.zeros_like(img)
        #     shifted_shifted_img[:, x_shift:] = img[:, :-x_shift]
        # elif x_shift < 0:
        #     shifted_shifted_img = np.zeros_like(img)
        #     shifted_shifted_img[:, :x_shift] = img[:, -x_shift:]
        # else:
        #     shifted_shifted_img = img.copy()  # No shift for x_shift = 0
        shifted_shifted_img = np.roll(img, x_shift, axis=1)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(img, cmap='gray')
        # plt.title("Original Image")
        # plt.subplot(1, 2, 2)
        # plt.imshow(shifted_shifted_img, cmap='gray')
        # plt.title("Shifted Image")
        # plt.show()
        # flatten for np.correlate
        if normalised:
            img_cent = img - np.mean(img)
            shifted_shifted_cent = shifted_shifted_img - np.mean(shifted_shifted_img)
            num = np.sum(img_cent*shifted_shifted_cent)
            den = np.sqrt(np.sum(img_cent**2)*np.sum(shifted_shifted_cent**2))
            cross_corr = num/den
        else:
            img_flat = img.flatten()
            shifted_shifted_flat = shifted_shifted_img.flatten()

            cross_corr = np.correlate(img_flat, shifted_shifted_flat, mode='valid')
            
        # add max value and shift value to list
        corr_values.append(np.max(cross_corr))
        shift_values.append(x_shift)

    corr_values = np.array(corr_values)
    shift_values = np.array(shift_values)
    

    return shift_values, corr_values