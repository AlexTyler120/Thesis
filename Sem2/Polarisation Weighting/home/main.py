import Images
import ShiftEstimate
import WeightingEstimate
import matplotlib.pyplot as plt
def main():
    RESIZE_VAR = 0.5
    GREY = False
    SIMULATED_SHIFT = 12
    WEIGHTING_SIM = 0.7

    ### Shift estimates with polarised images ###
    # img1, _, _ = Images.read_image("python/test_im/ball/ball_0.png", RESIZE_VAR, GREY)
    # img2, _, _ = Images.read_image("python/test_im/ball/ball_90.png", RESIZE_VAR, GREY)
    # transformed_image = Images.create_shifted_image_polarised_imgs(img1, img2, SIMULATED_SHIFT, True)
    ### ###

    ### Shift estimates with simulated images ###
    img, _, _ = Images.read_image("python/test_im/bird.jpg", RESIZE_VAR, GREY)
    transformed_image = Images.create_shifted_simulation(img, WEIGHTING_SIM, SIMULATED_SHIFT)
    ### ###

    shift_est = ShiftEstimate.compute_pixel_shift(transformed_image)
    print(f"Shift estimate: {shift_est}")
    w1_vals = []
    for i in range(3):
        img_channel = transformed_image[:, :, i]

        est_w1 = WeightingEstimate.optimise_psf(img_channel, shift_est)

        w1_vals.append(est_w1)
    print(w1_vals)
    plt.show()
if __name__ == "__main__":
    main()