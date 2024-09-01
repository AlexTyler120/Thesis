import ImageRun
def main():
    RESIZE_VAR = 0.2
    GREY = False
    SIMULATED_SHIFT = 6
    WEIGHTING_SIM = 0.7
    ANGLE = 0

    ### Shift estimates with polarised images ###
    transformed_image = ImageRun.polarised_generation("ball", ANGLE, RESIZE_VAR, GREY, SIMULATED_SHIFT)
    ### ###

    ### Shift estimates with simulated images ###
    # transformed_image = ImageRun.simulated_generation("ball_0.png", SIMULATED_SHIFT, RESIZE_VAR, GREY, WEIGHTING_SIM)
    ### ###

    ### Run estimation only getting w1 ###
    # ImageRun.run_estimate_w1(transformed_image)
    ### ###

    ### Run estimation getting w1 and w2 ###
    # ImageRun.run_estimate_w1_w2(transformed_image)
    ### ###

    ### Run estimation using all image channels ###
    # ImageRun.run_combined_optimisation(transformed_image)
    ### ###

    ### Run estimation but estimating for every single psf value ###
    # ImageRun.run_estimation_all_weights(transformed_image)
    ### ###

    ### Run estimation with local deconvolution ###
    ImageRun.run_local_patching(transformed_image)
    ### ###
if __name__ == "__main__":
    main()