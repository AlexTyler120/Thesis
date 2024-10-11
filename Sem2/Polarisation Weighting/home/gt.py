import numpy as np
import matplotlib.pyplot as plt
import cv2
def gt():
    # i0path = 'python/test_im/fakefruit/low_fakefruit_0.png'
    # i90path = 'python/test_im/fakefruit/low_fakefruit_90.png'
    # i135path = 'python/test_im/fakefruit/low_fakefruit_135.png'
    # i45path = 'python/test_im/fakefruit/low_fakefruit_45.png'
    i0path = 'python/test_im/fakefruit/rect_fakefruit_0.png'
    i90path = 'python/test_im/fakefruit/rect_fakefruit_90.png'
    i135path = 'python/test_im/fakefruit/rect_fakefruit_135.png'
    i45path = 'python/test_im/fakefruit/rect_fakefruit_45.png'
    # i0path = 'python/test_im/caligraphset/small_caligraphset_0.png'
    # i90path = 'python/test_im/caligraphset/small_caligraphset_90.png'
    # i135path = 'python/test_im/caligraphset/small_caligraphset_135.png'
    # i45path = 'python/test_im/caligraphset/small_caligraphset_45.png'
    I0 = cv2.imread(i0path)
    I45 = cv2.imread(i45path)
    I90 = cv2.imread(i90path)
    I135 = cv2.imread(i135path)

    channels = ['R', 'G', 'B']
    polarization_angles = {}
    dop_angles = {}
    polar_partial = {}
    dop_partial = {}
    ea_all = {}
    
    # for i in range(3):
    #     # normalise 0 - 1
    #     print(np.max(I0[:, :, i]))
    #     I0[:, :, i] = I0[:, :, i] / np.max(I0[:, :, i])
        
    #     I45[:, :, i] = I45[:, :, i] / np.max(I45[:, :, i])
    #     I90[:, :, i] = I90[:, :, i] / np.max(I90[:, :, i])
    #     I135[:, :, i] = I135[:, :, i] / np.max(I135[:, :, i])

    for i, color in enumerate(channels):
        # Extract each channel
        I0_channel = I0[:, :, i]
        I45_channel = I45[:, :, i] 
        I90_channel = I90[:, :, i] 
        I135_channel = I135[:, :, i]
        # Normalize the images using cv2's normalize function
        I0_channel = cv2.normalize(I0_channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        I45_channel = cv2.normalize(I45_channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        I90_channel = cv2.normalize(I90_channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        I135_channel = cv2.normalize(I135_channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        S0 = 0.5 * (I0_channel + I90_channel + I45_channel + I135_channel)
        S1 = I0_channel - I90_channel
        S2 = I45_channel - I135_channel
        S3 = I135_channel - I45_channel
        
        theta = 0.5*np.arctan2(S2, S1)
        theta_deg = np.degrees(theta) 
        
        # get dolp
        dolp = np.sqrt(S1**2 + S2**2) / (S0 + 1e-6)
        # clip
        dolp = np.clip(dolp, -1, 1)
        
        dop_angles[color] = dolp
        
        # Store the result in a dictionary
        polarization_angles[color] = np.mod(theta_deg, 180)
        
        diff = I0_channel - I90_channel
        sum = I0_channel + I90_channel
        
        dop_par = np.abs(diff) / (sum + 1e-6)
        # clip
        dop_par = np.clip(dop_par, -1, 1)
        aop_par = np.degrees(np.arctan2(np.abs(diff), sum))
        
        dop_partial[color] = dop_par

        polar_partial[color] = aop_par
    plt.figure()
    plt.imshow(I0_channel, cmap='Blues')
    plt.figure()
    plt.imshow(I90_channel, cmap='Reds')
    # stokes
# gt()