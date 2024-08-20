import Image
import shiftImage
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import skimage as sk
import plotly.graph_objects as go
import plotly.subplots as spgraph
import cv2

def compute_psf(shift, w1, w2):
    print(shift)
    psf = np.zeros(shift + 1)
    psf[-1] = w1
    psf[0] = w2

    psf = psf/np.sum(psf)

    psf = np.expand_dims(psf, axis=0)
    print(psf)
    return psf

def computeCrossCorrelation(img):
    max_shift = img.shape[1] // 2
    # max_shift = 26
    shift_vals = []
    corr_vals = []

    for x_shift in range(-max_shift, max_shift + 1):
        It_shifted = sp.ndimage.shift(img, shift=(0, x_shift), mode='constant', cval=0)

        # flatten
        It_flat = img.flatten()
        It_shifted_flat = It_shifted.flatten()

        cross_corr = np.correlate(It_flat, It_shifted_flat, mode="valid")

        corr_val = np.max(cross_corr)

        shift_vals.append(x_shift)
        corr_vals.append(corr_val)
    
    shift_vals = np.array(shift_vals)
    corr_vals = np.array(corr_vals)

    return shift_vals, corr_vals


# def loss_func(est, img, shift):
#     w1_est = est
#     w2_est = 1 - w1_est
    
#     # Scale img between 0 and 1
#     ## original blurred image
#     img = img / np.max(img)
#     # plt.figure()
#     # plt.imshow(img, cmap="gray")
    
#     # Compute PSF and perform Wiener deconvolution
#     psf = compute_psf(shift, w1_est, w2_est)
#     # deconvolved original blurred with estimated psf
#     deconvolved = sk.restoration.wiener(img, psf, balance=0)
#     # plt.figure()
#     # plt.imshow(deconvolved, cmap="gray")
#     # Generate shifted images
#     I1_est = deconvolved.copy()
#     I2_est = I1_est.copy()
#     I2_est[:, :-shift] = I2_est[:, shift:]

#     # Estimate It based on the estimated weights
#     # estimated blurred image from deconvolved with estimated psf
#     It_est = w1_est * I1_est + w2_est * I2_est
#     # plt.figure()
#     # plt.imshow(It_est, cmap="gray")
#     # plt.show()
#     # Compute cross-correlation of the estimated image with itself
#     shifts, corr = computeCrossCorrelation(deconvolved)
    
#     # Apply Savitzky-Golay filter to smooth the correlation and find peaks
#     window_length = 5  
#     polyorder = 3 
#     y_baseline = sp.signal.savgol_filter(corr, window_length, polyorder)
#     corr_filtered = corr - y_baseline

#     # Identify the indices of the peaks at the target shifts
#     target_shifts = [shift, -shift]
#     peak_loss = 0
#     for ts in target_shifts:
#         index = np.where(shifts == ts)[0]
#         if len(index) > 0:
#             peak_loss += np.abs(corr_filtered[index[0]])

#     # Compute cross-correlation using FFT
#     img_fft = np.fft.fft2(img)
#     It_est_fft = np.fft.fft2(It_est)
#     corr_fft = np.fft.ifft2(img_fft * np.conj(It_est_fft))
#     corr = np.abs(np.fft.fftshift(corr_fft))

#     # Combine the losses with appropriate weights
#     total_loss = - np.max(corr)
#     print(total_loss)
    
#     return total_loss

def computeCrossCorrelation(img):
    max_shift = img.shape[1] // 2
    shift_vals = []
    corr_vals = []
    
    for x_shift in range(-max_shift, max_shift + 1):
        It_shifted = sp.ndimage.shift(img, shift=(0, x_shift), mode='constant', cval=0)
        corr_val = np.correlate(img.flatten(), It_shifted.flatten(), mode="valid").max()
        shift_vals.append(x_shift)
        corr_vals.append(corr_val)
    
    return np.array(shift_vals), np.array(corr_vals)


def loss_func(est, img, shift):
    w1_est = est
    w2_est = 1 - w1_est
    
    # Scale img between 0 and 1
    img = img / np.max(img)
    
    # Compute PSF and perform Wiener deconvolution
    psf = compute_psf(shift, w1_est, w2_est)
    deconvolved = sk.restoration.wiener(img, psf, balance=0)

    # Autocorrelation of the deconvolved image
    deconvolved_fft = np.fft.fft2(deconvolved)
    autocorr_fft = np.fft.ifft2(deconvolved_fft * np.conj(deconvolved_fft))
    autocorr = np.abs(np.fft.fftshift(autocorr_fft))

    # Detect peaks in the autocorrelation
    shifts = np.arange(-autocorr.shape[1] // 2, autocorr.shape[1] // 2 + 1)
    center = autocorr.shape[1] // 2
    autocorr_line = autocorr[center, :]
    
    peaks, properties = sp.signal.find_peaks(autocorr_line, height=0.1, distance=shift, prominence=0.1)

    # Penalize for significant peaks indicating shifts
    shift_loss = 0
    for peak in peaks:
        if abs(shifts[peak]) == shift:
            shift_loss += properties['prominence'][peaks == peak][0]

    # Directly penalize shifts in the deconvolved image
    direct_shift_penalty = 0
    if len(peaks) > 0:
        direct_shift_penalty = min([abs(shifts[peak]) for peak in peaks])

    # Image difference loss (Mean Squared Error)
    mse_loss = np.mean((img - deconvolved) ** 2)

    # Adjusted penalty for extreme values of w1 or w2
    epsilon = 1e-10  # small value to avoid log(0)
    weight_penalty = 0.01  # Reduced weight to prevent it from dominating
    penalty = -weight_penalty * np.log(w1_est * (1 - w1_est) + epsilon)

    # Combine the losses with appropriate weights
    weight_shift = 10  # Increased weight for shift detection
    weight_mse = 1  # Normal weight for MSE
    weight_direct_shift = 5  # Additional penalty for direct shift detection

    total_loss = (weight_shift * shift_loss + 
                  weight_mse * mse_loss + 
                  weight_direct_shift * direct_shift_penalty + 
                  penalty)

    print(total_loss)
    return total_loss


def interactive_plots(deconvolved, shifted1, corr1, shifted2, corr2):
    # Interactive image display
    fig = go.Figure()
    
    fig.add_trace(go.Image(z=deconvolved))
    
    fig.update_layout(
        title="Deconvolved Image",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        coloraxis_showscale=False
    )
    
    fig.show()

    # Interactive correlation vs shift plots
    fig = spgraph.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           subplot_titles=("Correlation vs Shift (Set 1) Est", "Correlation vs Shift (Set 2) Real"))
    
    fig.add_trace(go.Scatter(x=shifted1, y=corr1, mode='lines', name='Set 1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=shifted2, y=corr2, mode='lines', name='Set 2'), row=2, col=1)
    
    fig.update_xaxes(title_text="Shift")
    fig.update_yaxes(title_text="Correlation")
    
    fig.update_layout(height=600, width=800, title_text="Interactive Correlation Plots")
    
    fig.show()


def main():
    path = "python/flower.jpg"
    size = 0.05
    grey = True

    original_image = Image.Image(path, size, grey)
    w1 = 0.7
    w2 = 0.3

    shift = 6

    shifted = shiftImage.shiftImage(original_image, w1, w2, shift)
    shifted.computePixelShift()

    w1guess = 0.3

    bounds = [(0, 1)]

    # loss_func(weight_guess, shifted.It, shift)
    result = sp.optimize.minimize(loss_func, w1guess, args=(shifted.It, shift), bounds=bounds, method='Powell') # Powell L-BFGS-B
    print(result.x)

    psf = compute_psf(shift, result.x, 1 - result.x)
    img = shifted.It/ np.max(shifted.It)
    deconvolved1 = sk.restoration.wiener(img, psf, balance=0)
    # corr1 = sp.signal.correlate2d(deconvolved, deconvolved, mode='same')
    shifted1, corr1 = computeCrossCorrelation(deconvolved1)
    window_length = 5  
    polyorder = 3 
    y_baseline = sp.signal.savgol_filter(corr1, window_length, polyorder)
    corr1 = corr1 - y_baseline
    plt.figure()
    plt.imshow(deconvolved1, cmap='gray')
    plt.title('Deconvolved Image est')


    psf = compute_psf(shift, w1, w2)
    img = shifted.It/ np.max(shifted.It)
    deconvolved2 = sk.restoration.wiener(img, psf, balance=0)
    plt.figure()
    plt.imshow(deconvolved2, cmap='gray')
    plt.title('Deconvolved Image real')

    shifted2, corr2 = computeCrossCorrelation(deconvolved2)
    window_length = 5  
    polyorder = 3 
    y_baseline = sp.signal.savgol_filter(corr2, window_length, polyorder)
    corr2 = corr2 - y_baseline

    # # print corr vals at shift
    # print(corr1[np.where(shifted1 == shift)], corr1[np.where(shifted1 == -shift)])
    # print(corr1[np.where(shifted1 == 0)])
    # print(corr2[np.where(shifted2 == shift)], corr2[np.where(shifted2 == -shift)])
    # print(corr2[np.where(shifted2 == 0)])

    plt.show()

    
if __name__ == "__main__":
    main()
