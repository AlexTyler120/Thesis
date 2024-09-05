import ImageRun
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Patches
def main():
    RESIZE_VAR = 0.04
    GREY = False
    SIMULATED_SHIFT = 4
    WEIGHTING_SIM = 0.7
    ANGLE = 0

    ### Shift estimates with polarised images ###
    # transformed_image = ImageRun.polarised_generation("fakefruit", ANGLE, RESIZE_VAR, GREY, SIMULATED_SHIFT)
    ### ###

    ### Shift estimates with simulated images ###
    transformed_image = ImageRun.simulated_generation("fakefruit_0.png", SIMULATED_SHIFT, RESIZE_VAR, GREY, WEIGHTING_SIM)
    ### ###

    ### Run estimation only getting w1 ###
    # ImageRun.run_estimate_w1(transformed_image)
    ### ###

    ### Run estimation getting w1 and w2 ###
    # ImageRun.run_estimate_w1_w2(transformed_image)
    ### ###
if __name__ == "__main__":
    main()