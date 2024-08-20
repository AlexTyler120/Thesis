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


def loss_func(est, img, shift):
    w1_est = est
    w2_est = 1 - w1_est

    

    # contrast enhance image
    img = cv2.equalizeHist((img).astype(np.uint8))
    # Scale img between 0 and 1
    img = img / np.max(img)

    # Compute PSF and perform Wiener deconvolution
    psf = compute_psf(shift, w1_est, w2_est)
    deconvolved = sk.restoration.wiener(img, psf, balance=0)
    I1_est = deconvolved
    I2_est = I1_est.copy()
    I2_est[:, :-shift] = I2_est[:, shift:]

    img = sp.ndimage.shift(img, shift=(0,-shift//2), mode='constant', cval=0)

    blur_rec = w1_est*I1_est + w2_est*I2_est
    # blur_rec = w1_est*deconvolved[:, shift:] + w2_est*deconvolved[:, :-shift]

    cross_corr = sp.signal.correlate2d(img, blur_rec, mode='valid')

    plt.imshow(deconvolved, cmap='gray')
    plt.show()

    
    return  -np.sum(cross_corr)


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
    size = 0.2
    grey = True

    original_image = Image.Image(path, size, grey)
    w1 = 0.7
    w2 = 0.3

    shift = 10

    shifted = shiftImage.shiftImage(original_image, w1, w2, shift)
    shifted.computePixelShift()

    w1guess = 0.7

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
    deconvolved1 = sk.restoration.wiener(img, psf, balance=0)
    shifted1, corr1 = computeCrossCorrelation(deconvolved1)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(deconvolved1, cmap='gray')
    plt.title('Deconvolved Image est')
    plt.subplot(1,2,2)
    plt.plot(shifted1, corr1)
    plt.title('Correlation vs Shift (Set 1) Est')
    plt.xlabel('Shift')
    plt.ylabel('Correlation')
    
    psf = compute_psf(shift, w1, w2)
    img = shifted.It/ np.max(shifted.It)
    deconvolved2 = sk.restoration.wiener(img, psf, balance=0)
    shifted2, corr2 = computeCrossCorrelation(deconvolved2)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(deconvolved2, cmap='gray')
    plt.title('Deconvolved Image real')

    plt.subplot(1, 2, 2)
    plt.plot(shifted2, corr2)
    plt.title('Correlation vs Shift (Set 2) Real')
    plt.xlabel('Shift')
    plt.ylabel('Correlation')

    plt.show()

    
if __name__ == "__main__":
    main()
