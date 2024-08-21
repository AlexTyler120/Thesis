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
    # max_shift = img.shape[1] // 2
    max_shift = 20
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

def recreate_blur(img, shift, w1, w2):
    I1 = img.copy()
    I2 = I1.copy()

    I2[:, :-shift] = I2[:, shift:]
    blur_rec = w1*I1 + w2*I2

    return blur_rec

def loss_func(est, img, shift):
    w1_est = est
    w2_est = 1 - w1_est

    # contrast enhance image
    # Scale img between 0 and 1
    img = img / np.max(img)

    # img = sp.ndimage.shift(img, shift=(0,-shift//2), mode='constant', cval=0)

    # Compute PSF and perform Wiener deconvolution
    psf = compute_psf(shift, w1_est, w2_est)
    deconvolved = sk.restoration.wiener(img, psf, balance=0)

    # Use your computeCrossCorrelation function
    shift_vals, corr_vals = computeCrossCorrelation(deconvolved)

    # generate blur
    blur_rec = recreate_blur(deconvolved, shift, w1_est, w2_est)

    # Find the index of the peak at zero shift
    zero_idx = np.where(shift_vals == 0)[0][0]

    loss = 0

    # Check for gradients on the left side (should be positive)
    for i in range(1, zero_idx):
        # print(f"i at shift: {i} at {shift_vals[i]} and corr_vals[i]: {corr_vals[i]} and corr_vals[i-1]: {corr_vals[i-1]}")
        if corr_vals[i] <= corr_vals[i-1]:  # Gradient should be positive, but it's not
            loss += abs(corr_vals[i] - corr_vals[i-1])

    # Check for gradients on the right side (should be negative)
    for i in range(zero_idx + 1, len(corr_vals)):
        if corr_vals[i] >= corr_vals[i-1]:  # Gradient should be negative, but it's not
            loss += abs(corr_vals[i] - corr_vals[i-1])



    winlen = 5
    poly = 3
    corr_smooth = sp.signal.savgol_filter(corr_vals, winlen, poly)

    sav_filtered_corr = corr_vals - corr_smooth

    # Find the indices corresponding to ±shift
    pos_shift_idx = np.where(shift_vals == shift)[0]
    neg_shift_idx = np.where(shift_vals == -shift)[0]

     # Define a target strongly negative value for ±shift
    target_negative_value = -10  # Adjust this target as needed

    # Penalize if the values at ±shift deviate from the target negative value
    if len(pos_shift_idx) > 0:
        loss += abs(sav_filtered_corr[pos_shift_idx[0]] - target_negative_value)

    if len(neg_shift_idx) > 0:
        loss += abs(sav_filtered_corr[neg_shift_idx[0]] - target_negative_value)

    # Penalize if the values immediately adjacent to ±shift are not close to zero
    if len(pos_shift_idx) > 0 and pos_shift_idx[0] + 1 < len(sav_filtered_corr):
        loss += abs(sav_filtered_corr[pos_shift_idx[0] + 1])  # Right of +shift
    if len(pos_shift_idx) > 0 and pos_shift_idx[0] - 1 >= 0:
        loss += abs(sav_filtered_corr[pos_shift_idx[0] - 1])  # Left of +shift

    if len(neg_shift_idx) > 0 and neg_shift_idx[0] + 1 < len(sav_filtered_corr):
        loss += abs(sav_filtered_corr[neg_shift_idx[0] + 1])  # Right of -shift
    if len(neg_shift_idx) > 0 and neg_shift_idx[0] - 1 >= 0:
        loss += abs(sav_filtered_corr[neg_shift_idx[0] - 1])  # Left of -shift

    # print corr vals at +-shift
    print(f"corr_vals at +shift: {sav_filtered_corr[pos_shift_idx[0]]}")

    print(f"loss: {loss}")
    #print mse
    print(f"mse: {np.mean((img - blur_rec)**2)*100}")
    return loss

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
    path = "python/nopol.jpg"
    size = 0.15
    grey = True

    original_image = Image.Image(path, size, grey)
    w1 = 0.3
    w2 = 0.7

    shift = 10

    shifted = shiftImage.shiftImage(original_image, w1, w2, shift)
    shifted.computePixelShift()

    w1guess = 0.5
    bounds = [(0, 1)]

    # loss_func(weight_guess, shifted.It, shift)
    result = sp.optimize.minimize(loss_func, w1guess, args=(shifted.It, shift), bounds=bounds, method='Powell') # Powell L-BFGS-B
    # result = sp.optimize.differential_evolution(loss_func, bounds, args=(shifted.It, shift))
    # simulation annealing
    # result = sp.optimize.dual_annealing(loss_func, bounds, args=(shifted.It, shift))
    # result = sp.optimize.basinhopping(loss_func, w1guess, niter=100, minimizer_kwargs={'args': (shifted.It, shift), 'bounds': bounds})
    print(result.x)

    psf = compute_psf(shift, result.x, 1 - result.x)
    img = shifted.It/ np.max(shifted.It)
    # img = sp.ndimage.shift(img, shift=(0,-shift//2), mode='constant', cval=0)
    deconvolved1 = sk.restoration.wiener(img, psf, balance=0)
    
    shifted1, corr1 = computeCrossCorrelation(deconvolved1)
    winlen = 5
    poly = 3
    corr1_smooth = sp.signal.savgol_filter(corr1, winlen, poly)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(deconvolved1, cmap='gray')
    plt.title('Deconvolved Image est')
    plt.subplot(1,2,2)
    plt.plot(shifted1, corr1-corr1_smooth)
    plt.title('Correlation vs Shift (Set 1) Est')
    plt.xlabel('Shift')
    plt.ylabel('Correlation')
    
    psf = compute_psf(shift, w1, w2)
    deconvolved2 = sk.restoration.wiener(img, psf, balance=0)
    shifted2, corr2 = computeCrossCorrelation(deconvolved2)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(deconvolved2, cmap='gray')
    plt.title('Deconvolved Image real')

    

    plt.subplot(1, 2, 2)
    corr2_smooth = sp.signal.savgol_filter(corr2, winlen, poly)
    plt.plot(shifted2, corr2-corr2_smooth)
    plt.title('Correlation vs Shift (Set 2) Real')
    plt.xlabel('Shift')
    plt.ylabel('Correlation')

    plt.show()

    
if __name__ == "__main__":
    main()
