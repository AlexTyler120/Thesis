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
from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity as ssim


def main():
    path1 = "python/test_im/bird.jpg"
    # path1 = "python/test_im/blackwhitesquare.jpg"
    # path1 = "python/test_im/flowers.jpg"
    # path1 = "python/test_im/squareblackwhite1.jpg"
    # path1 = "python/flower.jpg"
    # path1 = "python/test_im/stonk.jpg"
    # path1 = "python/test_im/birckback.png"
    size = 0.6
    
    grey = False

    w1 = 0.4
    w2 = 1-w1

    shift = 12

    birdhalf = Image.Image(path1, size, grey)

    shiftedBirdHalf = shiftImage.shiftImage(birdhalf, w1, w2, shift)
    shiftedBirdHalf.computePixelShift()


    
    w1_vals = []
    for i in range(3):
        # cross correlation for original image before any processing
        original_img = birdhalf.original_image[:, :, i]
        shift_val, corr_val = shiftedBirdHalf.computeCrossCorrelation(original_img)
        # savgol filter
        corr_filt = shiftedBirdHalf.applyFilter(corr_val)

        # cross correlation for shifted image
        shiftedimg = shiftedBirdHalf.It[:, :, i]
        shift_val_shifted, corr_val_shifted = shiftedBirdHalf.computeCrossCorrelation(shiftedimg)

        corr_shift_filter = shiftedBirdHalf.applyFilter(corr_val_shifted)
        # optimisation
        # x = shiftedBirdHalf.opt_minimise_weights(0.51, [(0, 1)], "Powell", shiftedimg)

        w1_estimate, loss_val, clarity_w1, clarity_w2 = shiftedBirdHalf.opt_minimise_weights(0.51, [(0, 1)], "Powell", shiftedimg)
        print(f"x0 = {w1_estimate[0]} and the loss is {loss_val} and best w1 is {shiftedBirdHalf.best_w1}")
        
        final_x = w1_estimate
        if clarity_w2 < clarity_w1:
            final_x = 1 - w1_estimate

        w1_vals.append(final_x)

    
    print(w1_vals)
    # if all vals near each other take mean else retun error
    print(f"std of vals {np.std(w1_vals)}")
    if np.std(w1_vals) < 0.1:
        w1_final = np.mean(w1_vals)
        print(f"Final mean is {w1_final}")
    else:
        print("Error in weights")
        return
    

    shiftedBirdHalf.getImagePSF(w1_final)
    
    print(f"estiamted shift{shiftedBirdHalf.estimated_shift}, estimated psf {shiftedBirdHalf.estimated_psf}")
    colour_deconvolved = []
    for i in range(3):
        shiftedimg = shiftedBirdHalf.It[:, :, i]
        # deconvolution and cross correlation of said decoonvolution
        deconvolved = shiftedBirdHalf.deconvolve(shiftedimg)
        shifted_val_dec, corr_val_dec = shiftedBirdHalf.computeCrossCorrelation(deconvolved)
        corr_dec_filter = shiftedBirdHalf.applyFilter(corr_val_dec)
        colour_deconvolved.append(deconvolved)
        plt.figure()
        plt.subplot(3, 3, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title("Original Bird")
        plt.subplot(3, 3, 2)
        plt.imshow(shiftedimg, cmap='gray')
        plt.title(f"Shifted Bird weights {w1} & {w2}")
        plt.subplot(3, 3, 3)
        plt.imshow(deconvolved, cmap='gray')
        plt.title(f"Deconvolved Bird PSF vals {shiftedBirdHalf.estimated_psf[0][-1]:.2f} & {shiftedBirdHalf.estimated_psf[0][0]:.2f}")

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
    stacked_image = np.dstack(colour_deconvolved)*255

    # Convert the image to uint8
    image_to_display = np.clip(stacked_image, 0, 255).astype(np.uint8)

    # Now apply cvtColor and display the image
    plt.figure()
    plt.imshow(cv2.cvtColor(image_to_display, cv2.COLOR_BGR2RGB))
    plt.title("Deconvolved Bird")
    plt.show()

if __name__ == "__main__":
    main()
