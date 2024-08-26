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


def main():
    path = "python/80.png"
    size = 0.5
    grey = False

    original_image = Image.Image(path, size, grey)
    #w1 = 0.3
    #w2 = 0.7

    #shift = 10

    #shifted = shiftImage.shiftImage(original_image, w1, w2, shift)
    #shifted.computePixelShift()

    # w1guess = 0.5
    # bounds = [(0, 1)]
    # method = "Powell"
    # shifted.opt_minimise_weights(w1guess, bounds, method)

    actual_shift = shiftImage.shiftImage(original_image, 0, 1, 0)
    actual_shift.computePixelShift()
    prev_corr = [0]
    for i in range(3):
        original_img = original_image.original_image[:,:, i]
        shift_val, corr_val = actual_shift.computeCrossCorrelation(original_img)
        corr_filt = actual_shift.applyFilter(corr_val)

        shiftedimg = actual_shift.It[:,:,i]
        shift_val_shifted, corr_val_shifted = actual_shift.computeCrossCorrelation(shiftedimg)
        if len(prev_corr) == 1:
            prev_corr = corr_val_shifted
        else:
            prev_corr = [a + b for a, b in zip(prev_corr, corr_val_shifted)]

        corr_shift_filter = actual_shift.applyFilter(corr_val_shifted)
        x = actual_shift.opt_minimise_weights(0.51, [(0,1)], "Powell", shiftedimg)

        actual_shift.getImagePSF(actual_shift.best_w1)

        print(f" x0 = {x[0]} and the loss is {actual_shift.min_loss} and best w1 is {actual_shift.best_w1}")
        print(actual_shift.estimated_shift)
        # deconvolution and cross correlation of said decoonvolution
        deconvolved = actual_shift.deconvolve(shiftedimg)
        shifted_val_dec, corr_val_dec = actual_shift.computeCrossCorrelation(deconvolved)
        corr_dec_filter = actual_shift.applyFilter(corr_val_dec)

        plt.figure()
        plt.subplot(3, 3, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title("Original Bird")
        plt.subplot(3, 3, 2)
        plt.imshow(shiftedimg, cmap='gray')
        plt.title(f"Shifted Bird weights ? & ?")
        plt.subplot(3, 3, 3)
        plt.imshow(deconvolved, cmap='gray')
        plt.title(f"Deconvolved Bird PSF vals {actual_shift.estimated_psf[0][-1]:.2f} & {actual_shift.estimated_psf[0][0]:.2f}")

        plt.subplot(3, 3, 4)
        plt.plot(shift_val, corr_val)
        plt.title("Cross Correlation for Bird")
        plt.subplot(3, 3, 5)
        plt.plot(shift_val_shifted, corr_val_shifted)
        plt.title("Cross Correlation for Shifted Bird")
        plt.subplot(3, 3, 6)
        plt.plot(shifted_val_dec, corr_val_dec)
        plt.title("Cross Correlation for Deconvolved Bird")

        plt.subplot(3, 3, 7)
        plt.plot(shift_val, corr_filt)
        plt.title("Filtered Cross Correlation for Bird")
        plt.subplot(3, 3, 8)
        plt.plot(shift_val_shifted, corr_shift_filter)
        plt.title("Filtered Cross Correlation for Shifted Bird")
        plt.subplot(3, 3, 9)
        plt.plot(shifted_val_dec, corr_dec_filter)
        plt.title("Filtered Cross Correlation for Deconvolved Bird")


        # plt.show()
    plt.figure()
    plt.plot(shift_val_shifted, prev_corr)
    plt.title("Cross Correlation for Shifted Bird")
    plt.show()

    print(actual_shift.best_w1)
if __name__ == "__main__":
    main()
