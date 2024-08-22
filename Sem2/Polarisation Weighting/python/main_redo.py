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
    # path1 = "python/nopol.jpg"
    size = 0.5
    
    grey = False

    w1 = 0.2
    w2 = 0.8

    shift = 10

    birdhalf = Image.Image(path1, size, grey)
    # blackwhitesquarehalf = Image.Image(path2, size, grey)
    # flowershalf = Image.Image(path3, size, grey)
    # squareblackwhite1half = Image.Image(path4, size, grey)

    shiftedBirdHalf = shiftImage.shiftImage(birdhalf, w1, w2, shift)
    shiftedBirdHalf.computePixelShift()

    # shiftedBlackWhiteSquareHalf = shiftImage.shiftImage(blackwhitesquarehalf, w1, w2, shift)
    # shiftedBlackWhiteSquareHalf.computePixelShift()

    # shiftedFlowersHalf = shiftImage.shiftImage(flowershalf, w1, w2, shift)
    # shiftedFlowersHalf.computePixelShift()

    # shiftedSquareBlackWhite1Half = shiftImage.shiftImage(squareblackwhite1half, w1, w2, shift)
    # shiftedSquareBlackWhite1Half.computePixelShift()


    # plot birdhalf, shifted birdhalf, cross corr for birdhalf and crosscorr for shiftedbirdhalf
    for i in range(3):
        original_img = birdhalf.original_image[:, :, i]
        shift_val, corr_val = shiftedBirdHalf.computeCrossCorrelation(original_img)

        corr_filt = shiftedBirdHalf.applyFilter(corr_val)

        shiftedimg = shiftedBirdHalf.It[:, :, i]
        shift_val_shifted, corr_val_shifted = shiftedBirdHalf.computeCrossCorrelation(shiftedimg)
        corr_shift_filter = shiftedBirdHalf.applyFilter(corr_val_shifted)
        x = shiftedBirdHalf.opt_minimise_weights(0.51, [(0, 1)], "Powell", shiftedimg)

        shiftedBirdHalf.getImagePSF(w1)
        print(shiftedBirdHalf.estimated_shift)

        deconvolved = shiftedBirdHalf.deconvolve(shiftedimg)
        shifted_val_dec, corr_val_dec = shiftedBirdHalf.computeCrossCorrelation(deconvolved)
        corr_dec_filter = shiftedBirdHalf.applyFilter(corr_val_dec)

        plt.figure()
        plt.subplot(3, 3, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title("Original Bird")
        plt.subplot(3, 3, 2)
        plt.imshow(shiftedimg, cmap='gray')
        plt.title("Shifted Bird")
        plt.subplot(3, 3, 3)
        plt.imshow(deconvolved, cmap='gray')
        plt.title("Deconvolved Bird")

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

        plt.show()
    plt.show()

    print(shiftedBirdHalf.best_w1)

if __name__ == "__main__":
    main()
