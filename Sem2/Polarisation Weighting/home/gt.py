import numpy as np
import matplotlib.pyplot as plt
import cv2

def calc_stokes(I0, I45, I90, I135):
    S0 = 0.5 * (I0 + I90 + I45 + I135)
    S1 = I0 - I90
    S2 = I45 - I135
    S3 = I135 - I45
    return S0, S1, S2, S3

def calc_dolp(S0, S1, S2):
    s0_ = np.clip(S0, 1e-6, None)
    dolp = np.sqrt(S1**2 + S2**2) / s0_
    return np.clip(dolp, 0, 1)

def calc_aolp(S1, S2):
    s1_ = np.copy(S1)
    s1_[S1 == 0] = 1e-6
    phase = np.rad2deg(np.arctan2(S2, s1_))
    mask = (phase < 0)
    phase[mask] = phase[mask] + 180*2
    phase /= 2
    return np.clip(phase, 0, 180)

def gt():
    # i0path = 'python/test_im/fakefruit/low_fakefruit_0.png'
    # i90path = 'python/test_im/fakefruit/low_fakefruit_90.png'
    # i135path = 'python/test_im/fakefruit/low_fakefruit_135.png'
    # i45path = 'python/test_im/fakefruit/low_fakefruit_45.png'
    # i0path = 'python/test_im/fakefruit/sq_fakefruit_0.png'
    # i90path = 'python/test_im/fakefruit/sq_fakefruit_90.png'
    # i135path = 'python/test_im/fakefruit/sq_fakefruit_135.png'
    # i45path = 'python/test_im/fakefruit/sq_fakefruit_45.png'
    # i0path = 'python/test_im/fakefruit/rect_fakefruit_0.png'
    # i90path = 'python/test_im/fakefruit/rect_fakefruit_90.png'
    # i135path = 'python/test_im/fakefruit/rect_fakefruit_135.png'
    # i45path = 'python/test_im/fakefruit/rect_fakefruit_45.png'
    i0path = 'python/test_im/dragon/small_dragon_0.png'
    i90path = 'python/test_im/dragon/small_dragon_90.png'
    i135path = 'python/test_im/dragon/small_dragon_135.png'
    i45path = 'python/test_im/dragon/small_dragon_45.png'
    I0 = cv2.imread(i0path)
    # bgr to rgb
    I0 = I0[:, :, ::-1]
    I45 = cv2.imread(i45path)
    #bgr to rgb
    I45 = I45[:, :, ::-1]
    I90 = cv2.imread(i90path)
    # bgr to rgb
    I90 = I90[:, :, ::-1]
    I135 = cv2.imread(i135path)
    # bgr to rgb
    I135 = I135[:, :, ::-1]

    channels = ['R', 'G', 'B']
    polarization_angles = {}
    dop_angles = {}
    polar_partial = {}
    dop_partial = {}
    ea_all = {}
    
    

    for i, color in enumerate(channels):
        # Extract each channel
        I0_channel = I0[:, :, i]
        I45_channel = I45[:, :, i] 
        I90_channel = I90[:, :, i] 
        I135_channel = I135[:, :, i]

        # Normalize the images by the global max and min
        global_min = min(I0_channel.min(), I45_channel.min(), I90_channel.min(), I135_channel.min())
        global_max = max(I0_channel.max(), I45_channel.max(), I90_channel.max(), I135_channel.max())

        I0_channel = (I0_channel - global_min) / (global_max - global_min)
        I45_channel = (I45_channel - global_min) / (global_max - global_min)
        I90_channel = (I90_channel - global_min) / (global_max - global_min)
        I135_channel = (I135_channel - global_min) / (global_max - global_min)

        # Normalize the images using cv2's normalize function

        # Calculate the Stokes parameters
        S0, S1, S2, S3 = calc_stokes(I0_channel, I45_channel, I90_channel, I135_channel)
        theta_deg = calc_aolp(S1, S2)
        dolp = calc_dolp(S0, S1, S2)
        
        dop_angles[color] = dolp
        
        # Store the result in a dictionary
        polarization_angles[color] = theta_deg

        break
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(polarization_angles['R'], cmap='jet')
    plt.title('AoLP')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(dop_angles['R'], cmap='jet')
    plt.title('DoLP')
    plt.figure()
    plt.imshow(np.abs(I0_channel - I90_channel), cmap='jet')
    plt.title('I0 - I90')
    plt.axis('off')
    # plt.title('Polarization Angle Red')
    # plt.colorbar()
    # plt.axis('off')
    # plt.figure()
    # plt.imshow(polarization_angles['G'], cmap='jet')
    # plt.title('Polarization Angle')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(polarization_angles['B'], cmap='jet')
    # plt.colorbar()
    # plt.title('Polarization Angle')
    
    # plt.figure()
    # plt.imshow(dop_angles['R'], cmap='jet')
    # plt.title('DOLP Red')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(dop_angles['G'], cmap='jet')
    
    # plt.title('DOLP Green')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(dop_angles['B'], cmap='jet')
    # plt.title('DOLP Blue')
    # plt.colorbar()
    
    
    
# gt()
# gt()
# plt.show()