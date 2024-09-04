import numpy as np
import Autocorrelation as ac
import matplotlib.pyplot as plt

def obtain_correlation_peaks(shift_vals, corr_vals):
    """
    Obtain the peaks of the correlation values
    shift_vals: the shift values
    corr_vals: the correlation values
    """
    peaks = []
    steepness = []

    for i in range(len(shift_vals) - 1):
        zero_idx = np.where(shift_vals == 0)[0][0]
        if i == 0 or i == len(shift_vals) - 1 or (i >= zero_idx - 3 and i <= zero_idx + 3):
            continue
        else:
            left_corr = corr_vals[i-1]
            right_corr = corr_vals[i+1]
            mid_corr = corr_vals[i]

            left_rise = mid_corr - left_corr
            right_rise = mid_corr - right_corr

            if left_rise > 0 and right_rise > 0:
                peaks.append(i)
                steepness.append(left_rise + right_rise)

    return peaks, steepness

def sort_peaks(peaks, steepness, shift_vals):
    """
    Sort the peaks in descending order of steepness
    peaks: the peaks of the correlation values
    steepness: the steepness of the peaks
    shift_vals: the shift values
    """
    sorted_peaks = [x for _, x in sorted(zip(steepness,peaks), reverse=True)]
    # get top 3 peaks
    top_peaks = sorted_peaks[:2]
    peak_shifts = [shift_vals[peak] for peak in top_peaks]

    if abs(peak_shifts[0]) == abs(peak_shifts[1]):
        estimated_shift = abs(peak_shifts[0])
    else:
        print("Error multiple peaks detected or no peaks detected")
        estimated_shift = 2

    return estimated_shift

def compute_pixel_shift(img):
    """
    Compute the pixel shift of an image using the autocorrelation
    img: the image to compute the pixel shift of
    """
    est_shifts = []
    for i in range(img.shape[2]):
        # normalise the image
        img_norm = (img[:, :, i] - np.mean(img[:, :, i])) / np.std(img[:, :, i])
        # compute the autocorrelation
        shift_vals, corr_vals = ac.compute_auto_corr(img_norm, None, True)
        # obtain the peaks of the correlation
        filted_corr_vals = ac.obtain_peak_highlighted_curve(corr_vals)
        plt.figure()
        plt.plot(shift_vals, filted_corr_vals)
        plt.show()
        # obtain the peaks of the correlation
        peaks, steepness = obtain_correlation_peaks(shift_vals, filted_corr_vals)

        estiamted_shift = sort_peaks(peaks, steepness, shift_vals)

        est_shifts.append(estiamted_shift)

    estimate_shift = int(np.mean(est_shifts))

    return estimate_shift
