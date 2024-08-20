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
    # max_shift = img.image_width // 2
    max_shift = 26
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


def loss_func(est, img, shift):
    w1_est = est
    w2_est = 1 - w1_est
    
    # Scale img between 0 and 1
    img = img / np.max(img)
    
    # Compute PSF and perform Wiener deconvolution
    psf = compute_psf(shift, w1_est, w2_est)
    deconvolved = sk.restoration.wiener(img, psf, balance=0)

    # Generate shifted images
    I1_est = deconvolved.copy()
    I2_est = I1_est.copy()
    I2_est[:, :-shift] = I2_est[:, shift:]
    It_est = w1_est * I1_est + w2_est * I2_est

    shifts, corr = computeCrossCorrelation(deconvolved)
    window_length = 5  
    polyorder = 3 
    y_baseline = sp.signal.savgol_filter(corr, window_length, polyorder)
    corr_filtered = corr - y_baseline

    # do for estimated
    shifts_est, corr_est = computeCrossCorrelation(It_est)
    y_baseline_est = sp.signal.savgol_filter(corr_est, window_length, polyorder)
    corr_filtered_est = corr_est - y_baseline_est

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
    size = 0.15
    grey = True

    original_image = Image.Image(path, size, grey)
    w1 = 0.3
    w2 = 0.7

    shift = 12

    shifted = shiftImage.shiftImage(original_image, w1, w2, shift)
    shifted.computePixelShift()

    w1guess = 0.5

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

    deconvolvedrl = sk.restoration.richardson_lucy(img, psf, num_iter=30)
    plt.figure()
    plt.imshow(deconvolvedrl, cmap='gray')
    plt.title('Deconvolved Image richard')
    plt.show()
    # corr1 = sp.signal.correlate2d(deconvolved, deconvolved, mode='same')
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

    
if __name__ == "__main__":
    main()
