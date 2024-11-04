import numpy as np
import matplotlib.pyplot as plt
import cv2

def calc_stokes(I0, I45, I90, I135):
    # S0 = 0.5*(I0 + I90 + I45 + I135)
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135
    S3 = I135 - I45
    return S0, S1, S2, S3

def calc_dolp(S0, S1, S2):
    # s0_ = np.clip(S0, 1e-6, None)
    
    dolp = np.sqrt(S1**2 + S2**2) / S0
    return np.clip(dolp, 0, 1)

def calc_aolp(S0, S1, S2):
    s1_ = np.copy(S1)
    
    s1_[S1 == 0] = 1e-6
    u = S2#/S0
    q = s1_#/S0
    phase = np.abs(np.degrees(np.arctan2(u, q)))
    mask = (phase < 0)
    phase[mask] = phase[mask] + 180*2
    phase /= 2
    # phase = 0.5 * np.arctan2(S2, S1)
    # phase = np.degrees(phase)
    return phase

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
    prefix = "small_"
    item = "caligraphset"
    i0path = 'python/test_im/'+item+'/'+prefix+item+'_0.png'
    i90path = 'python/test_im/'+item+'/'+prefix+item+'_90.png'
    i135path = 'python/test_im/'+item+'/'+prefix+item+'_135.png'
    i45path = 'python/test_im/'+item+'/'+prefix+item+'_45.png'
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
    
    S0_total = np.zeros_like(I0[:, :, 0], dtype=np.float32)
    S1_total = np.zeros_like(I0[:, :, 0], dtype=np.float32)
    S2_total = np.zeros_like(I0[:, :, 0], dtype=np.float32)
    for i, color in enumerate(channels):
        # Extract each channel
        I0_channel = I0[:, :, i]
        I45_channel = I45[:, :, i] 
        I90_channel = I90[:, :, i] 
        I135_channel = I135[:, :, i]

        I0_channel = cv2.normalize(I0_channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        I45_channel = cv2.normalize(I45_channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        I90_channel = cv2.normalize(I90_channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        I135_channel = cv2.normalize(I135_channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Normalize the images using cv2's normalize function

        # Calculate the Stokes parameters
        S0, S1, S2, S3 = calc_stokes(I0_channel, I45_channel, I90_channel, I135_channel)
        theta_deg = calc_aolp(S0, S1, S2)
        dolp = calc_dolp(S0, S1, S2)
        
        dop_angles[color] = dolp
        
        # Store the result in a dictionary
        polarization_angles[color] = theta_deg
        print(color)
        
        # Accumulate the Stokes parameters
        S0_total += S0
        S1_total += S1
        S2_total += S2

    # Calculate composite AoLP and DoLP
    composite_dolp = calc_dolp(S0_total, S1_total, S2_total)
    composite_aolp = calc_aolp(S0_total, S1_total, S2_total)
    # plt.figure()
    # # plt.subplot(1,2,1)
    # # plt.imshow(np.clip(polarization_angles['R'], -55, 25), cmap='jet')
    # plt.imshow(np.clip(polarization_angles['B'], 0, 90), cmap='jet')
    # plt.title('AoLP')
    # plt.colorbar()
    # plt.axis('off')
    # plt.figure()
    # plt.imshow(np.clip(dop_angles['B'], 0, 1), cmap='jet')
    # plt.title('DoLP')
    # plt.colorbar()
    # plt.axis('off')
    # plt.subplot(1,2,2)
    # plt.imshow(np.clip(dop_angles['B'], 0, 1), cmap='jet')
    # plt.title('DoLP')
    # plt.colorbar()
    # plt.figure()
    # difference = I0_channel - I90_channel
    # vmax = np.max(np.abs(difference))
    # plt.imshow(difference, cmap='jet', vmin=-vmax, vmax=vmax)
    # plt.title('I0 - I90')
    # plt.colorbar()
    # plt.axis('off')
    # plt.figure()
    # plt.imshow(I0_channel, cmap="jet")
    # plt.title('I0')
    # plt.axis('off')
    
    I0_height = I0_channel.shape[0]
    I0_width = I0_channel.shape[1]
    w1_gt = np.zeros((I0_height, I0_width))
    w2_gt = np.zeros((I0_height, I0_width))
    
    for i in range(I0_height):
        for j in range(I0_width):
            I0_int = I0_channel[i, j]
            I90_int = I90_channel[i, j]
            int_var = [I0_int, I90_int]
            int_var = int_var / np.sum(int_var)
            w1_gt[i, j] = int_var[1]
            w2_gt[i, j] = int_var[0]
            
    s0, s1, s2, s3 = calc_stokes(I0_channel, I45_channel, I90_channel, I135_channel)
    
    w1_gt = (s0 + s1) / (2 * s0)
    w2_gt = (s0 - s1) / (2 * s0)
            
    angle = np.arctan2(w2_gt, w1_gt)
    angle = np.rad2deg(angle)
    # angle = np.clip(angle, 38, 50)
    # normalise 0 - 90
    # angle = (angle - np.min(angle)) / (np.max(angle) - np.min(angle))
    mag = np.sqrt(w1_gt**2 + w2_gt**2)
    # plt.figure(figsize=(12, 6))
    # # plt.subplot(1, 2, 1)
    # plt.imshow(I0)  # , cmap='jet')
    # # plt.title('Original Image')
    # plt.axis('off')
    # # plt.colorbar()
    # plt.tight_layout()
    # plt.figure()
    # # plt.subplot(1, 2, 2)
    # im = plt.imshow(composite_dolp, cmap='jet')
    # # plt.title('DoLP GT')
    # plt.axis('off')
    # plt.colorbar(im)
    # plt.tight_layout()
    # plt.figure()
    # # plt.subplot(1, 2, 2)
    # plt.imshow(composite_aolp, cmap='seismic')
    # # plt.title('AoLP GT')
    # plt.axis('off')
    # plt.colorbar()
    return polarization_angles, dop_angles, composite_aolp, composite_dolp
    
    
    
# gt()
# plt.show()