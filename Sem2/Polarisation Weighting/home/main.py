import Images
import ShiftEstimate
import WeightingEstimate
import matplotlib.pyplot as plt
from skimage import restoration as sk
import Autocorrelation as ac
import numpy as np
def main():
    RESIZE_VAR = 0.6
    GREY = False
    SIMULATED_SHIFT = 12
    WEIGHTING_SIM = 0.7

    ### Shift estimates with polarised images ###
    img1, _, _ = Images.read_image("python/test_im/ball/ball_0.png", RESIZE_VAR, GREY)
    img2, _, _ = Images.read_image("python/test_im/ball/ball_90.png", RESIZE_VAR, GREY)
    transformed_image = Images.create_shifted_image_polarised_imgs(img1, img2, SIMULATED_SHIFT, True)
    ### ###

    ### Shift estimates with simulated images ###
    # img, _, _ = Images.read_image("python/test_im/flowers.jpg", RESIZE_VAR, GREY)
    # transformed_image = Images.create_shifted_simulation(img, WEIGHTING_SIM, SIMULATED_SHIFT)
    ### ###

    shift_est = ShiftEstimate.compute_pixel_shift(transformed_image)
    print(f"Shift estimate: {shift_est}")
    w1_vals = []
    for i in range(3):
        img_channel = transformed_image[:, :, i]

        # est_w1 = WeightingEstimate.optimise_psf(img_channel, shift_est)
        est = WeightingEstimate.optimise_psf_both_weight(img_channel, shift_est)

        w1_vals.append(est)

    for i in range(len(w1_vals)):
        print(f"Channel {i}: {w1_vals[i]}")
        plt.figure()
        plt.subplot(1, 3, 1)
        image = transformed_image[:, :, i] / np.max(transformed_image[:, :, i])
        deconvolved = sk.wiener(image, WeightingEstimate.get_img_psf_w1_w2(w1_vals[i][0], w1_vals[i][1], shift_est), balance=0)
        # deconvolved between 0 -255
        plt.imshow(deconvolved, cmap='gray')
        plt.title(f"Channel {i} Deconvolved")
        plt.subplot(1, 3, 2)
        shift, corr = ac.compute_auto_corr(deconvolved, shift_est)
        plt.plot(shift, corr)
        plt.title(f"Channel {i} Auto Correlation")
        plt.subplot(1, 3, 3)
        plt.plot(shift, ac.obtain_peak_highlighted_curve(corr))
        plt.title(f"Channel {i} Filtered Auto Correlation")
        
    plt.show()
if __name__ == "__main__":
    main()