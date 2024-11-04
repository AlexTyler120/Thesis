import Autocorrelation as ac
import cv2
import pickle
import matplotlib.pyplot as plt
import ShiftEstimate as se
def main():
    image = cv2.imread("python/test_im/image.png")
    # convert to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image resize
    image = cv2.resize(image, (0,0), fx=1, fy=1)
    
    # cross correlation
    # image = image[:, :, 0]
    # shift, corr = ac.compute_auto_corr(image, 0, shift_est_func=True, patch_size=)
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.plot(shift, corr)
    # plt.subplot(1,2,2)
    # plt.imshow(image)
    
    image_0 = image.copy()
    image_1 = image.copy()
    image_1[:, 10:] = image_1[:, :-10]
    
    # combine
    combined = cv2.addWeighted(image_0, 0.5, image_1, 0.5, 0)
    shift_b, corr_b = ac.compute_auto_corr(combined[:,:, 2], 0, shift_est_func=True)
    shift_g, corr_g = ac.compute_auto_corr(combined[:,:, 1], 0, shift_est_func=True)
    shift_r, corr_r = ac.compute_auto_corr(combined[:,:, 0], 0, shift_est_func=True)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(shift_b, corr_b, color='b', label='Blue')
    plt.plot(shift_g, corr_g, color='g', label='Green')
    plt.plot(shift_r, corr_r, color='r', label='Red')
    # clip between -40 and 40
    # plt.xlim(-40, 40)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid('on')
    plt.legend(fontsize='20')
    plt.subplot(1,2,2)
    plt.imshow(combined)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()