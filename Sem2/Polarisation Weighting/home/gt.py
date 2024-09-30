import numpy as np
import matplotlib.pyplot as plt
import cv2
def gt():
    i0path = 'python/test_im/fakefruit/sq_fakefruit_0.png'
    i90path = 'python/test_im/fakefruit/sq_fakefruit_90.png'
    i135path = 'python/test_im/fakefruit/sq_fakefruit_135.png'
    i45path = 'python/test_im/fakefruit/sq_fakefruit_45.png'

    I0 = cv2.imread(i0path)
    I45 = cv2.imread(i45path)
    I90 = cv2.imread(i90path)
    I135 = cv2.imread(i135path)

    channels = ['R', 'G', 'B']
    polarization_angles = {}

    for i, color in enumerate(channels):
        # Extract each channel
        I0_channel = I0[:, :, i]
        I45_channel = I45[:, :, i]
        I90_channel = I90[:, :, i]
        I135_channel = I135[:, :, i]
        
        # stokes
        S0 = I0_channel + I90_channel
        S1 = I0_channel - I90_channel
        S2 = I45_channel - I135_channel
        
        theta = 0.5*np.arctan2(S2, S1)
        theta_deg = np.degrees(theta) 
        
        # Store the result in a dictionary
        polarization_angles[color] = theta_deg

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Red channel
    axs[0].imshow(polarization_angles['R'], cmap='jet')
    axs[0].set_title('Ground Truth Polarization Angle Heat Map - Red Channel')
    axs[0].axis('off')

    # Green channel
    axs[1].imshow(polarization_angles['G'], cmap='jet')
    axs[1].set_title('Polarization Angle Heat Map - Green Channel')
    axs[1].axis('off')

    # Blue channel
    axs[2].imshow(polarization_angles['B'], cmap='jet')
    axs[2].set_title('Polarization Angle Heat Map - Blue Channel')
    axs[2].axis('off')

    # Add colorbars
    for ax in axs:
        fig.colorbar(ax.images[0], ax=ax, orientation='vertical')

    # plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(polarization_angles['R'], cmap='jet')
    plt.title('Ground Truth - Red Channel')
    plt.axis('off')
    plt.colorbar(orientation='vertical')
    # plt.show()