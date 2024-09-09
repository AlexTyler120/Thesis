import ImageRun
import PatchRun
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

def main():
    RESIZE_VAR = 0.3
    GREY = False
    SIMULATED_SHIFT = 5
    WEIGHTING_SIM = 0.7
    ANGLE = 0
    PATCH_SIZE = 40

    ### Shift estimates with polarised images ###
    transformed_image = ImageRun.polarised_generation("pomegranate", ANGLE, RESIZE_VAR, GREY, SIMULATED_SHIFT)
    ### ###

    ### Shift estimates with simulated images ###
    # transformed_image = ImageRun.simulated_generation("pomegranate_0.png", SIMULATED_SHIFT, RESIZE_VAR, GREY, WEIGHTING_SIM)
    ### ###

    ### Run estimation only getting w1 ###
    # ImageRun.run_estimate_w1(transformed_image)
    ### ###

    ### Run estimation getting w1 and w2 ###
    # ImageRun.run_estimate_w1_w2(transformed_image)
    ### ###
    
    rgb, r, g, b = PatchRun.process_all_chanels(transformed_image, PATCH_SIZE)

    plt.figure(figsize=(10, 10))
    plt.subplot(2,2,1)
    plt.imshow(r*255, cmap='gray')
    plt.title("Red Channel")
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.imshow(g*255, cmap='gray')
    plt.title("Green Channel")
    plt.axis('off')
    
    plt.subplot(2,2,3)
    plt.imshow(b*255, cmap='gray')
    plt.title("Blue Channel")
    plt.axis('off')
    
    plt.subplot(2,2,4)
    plt.imshow(rgb)
    plt.title("RGB Image")
    plt.axis('off')
    
    plt.show()
    
    
if __name__ == "__main__":
    main()