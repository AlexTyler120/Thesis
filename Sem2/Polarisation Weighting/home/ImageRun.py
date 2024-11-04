import Images
import ShiftEstimate
import WeightingEstimate
import Autocorrelation as ac
import matplotlib.pyplot as plt
import numpy as np
from skimage import restoration as sk
import cv2
import pickle
import scipy as sp

def polarised_generation(file_name, prefix, degree, resize_var, grey, shift):
    """
    Generate a shifted image using two differently polarised images
    At least orthoganal to each other
    file_name: the file name of the first image
    degree: the degree of rotation of the second image
    resize_var: the amount to resize the image by
    grey: if True then convert the image to greyscale
    shift: the amount to shift the image by
    """
    path1 = "python/test_im/" + file_name +"/"+ prefix + file_name + "_"+str(degree)+".png"
    path2 = "python/test_im/" + file_name+"/" + prefix + file_name + "_"+str(degree+90)+".png"
    print(path1)
    print(path2)
    img1, _, _ = Images.read_image(path1, resize_var, grey)
    
    img2, _, _ = Images.read_image(path2, resize_var, grey)
    
    plt.figure()
    diff = np.abs(img1[:,:,0] - img2[:, :, 0]) + np.abs(img1[:,:,1] - img2[:, :, 1]) + np.abs(img1[:,:,2] - img2[:, :, 2])
    # normalise
    diff = diff / np.max(diff)
    plt.imshow(diff, cmap = "jet")
    plt.axis('off')
    # plt.colorbar()
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Intensity Difference', fontsize=16)
    plt.show()

    transformed_image = Images.create_shifted_image_polarised_imgs(img1, img2, shift, True)
    
    # transformed_image = Images.create_shifted_image_polarised_y(img1, img2, shift, True)
    return transformed_image

def simulated_generation(file_name, shift, resize_var, grey, weighting):
    """
    Generate a shifted image using a single image
    file_name: the file name of the image
    shift: the amount to shift the image by
    resize_var: the amount to resize the image by
    grey: if True then convert the image to greyscale
    weighting: the weighting to apply to the shift
    """

    path = "python/test_im/" + file_name

    img, _, _ = Images.read_image(path, resize_var, grey)
    plt.figure()
      # Increase brightness
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    transformed_image = Images.create_shifted_simulation(img, weighting, shift)

    return transformed_image


def run_estimate_w1(transformed_image):
    """
    Run estimation only getting w1
    transformed_image: the image to estimate the weighting for
    """

    shift_estimation = ShiftEstimate.compute_pixel_shift(transformed_image)
    w1_vals = []

    print(f"Shift estimate: {shift_estimation}")
    
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    for i in range(3):
        img_channel = transformed_image[:, :, i]

        est, _, all_losses, w1_vals_graph = WeightingEstimate.optimise_psf(img_channel, shift_estimation)
        
        with open(f'loss_values_{i}_powell_03.pkl', 'wb') as f:
            pickle.dump(all_losses, f)
            
        w1_vals.append(1-est)
        
        colors = ['red', 'green', 'blue']
        ax.plot(np.arange(1, len(all_losses) + 1, 1), all_losses, label=f"Channel {i}", color=colors[i])
        ax.set_title("Loss Values")
        ax.grid('on')
        ax2.plot(np.arange(1, len(w1_vals_graph) + 1, 1), w1_vals_graph, label=f"Channel {i}", color=colors[i])
        ax2.set_title("W1 Values")
        ax2.grid('on')
        break
    
    ax.legend()
    ax2.legend()
        
    dec = []
    shift = []
    corr = []
    for i in range(len(w1_vals)):
        print(f"Channel {i}: {w1_vals[i]}")
        # plt.figure()
        # plt.subplot(2, 2, i+1)

        # dec.append(sk.wiener(transformed_image[:, :, i], WeightingEstimate.get_img_psf(w1_vals[i], shift_estimation), balance=0))
        dec.append(WeightingEstimate.deconvolve_img(transformed_image[:, :, i], WeightingEstimate.get_img_psf(w1_vals[i], shift_estimation)))
        shift_, corr_ = ac.compute_auto_corr(dec[i], shift_estimation)
        shift.append(shift_)
        corr.append(corr_)
        # plt.imshow(deconvolved, cmap='gray')
        # plt.title(f"Channel {i} Deconvolved")
        # plt.axis('off')
        
        # plt.subplot(2, 2, i+2)
        # shift, corr = ac.compute_auto_corr(deconvolved, shift_estimation)
        # plt.plot(shift, corr)
        # plt.title(f"Channel {i} Auto Correlation")
        break
    plt.figure()
    # plt.subplot(2, 2, 1)
    plt.imshow(dec[0], cmap='gray', vmin=0, vmax=1)
    plt.title("Red Channel Deconvolved")
    plt.axis('off')
    
    # plt.subplot(2, 2, 2)
    # plt.imshow(dec[1], cmap='gray', vmin=0, vmax=1)
    # plt.title("Green Channel Deconvolved")
    # plt.axis('off')
    
    # plt.subplot(2, 2, 3)
    # plt.imshow(dec[2], cmap='gray', vmin=0, vmax=1)
    # plt.title("Blue Channel Deconvolved")
    # plt.axis('off')
    
    # plt.subplot(2, 2, 4)
    # plt.plot(shift[0], corr[0], label="Red", color='red')
    # plt.plot(shift[1], corr[1], label="Green", color = 'green')
    # plt.plot(shift[2], corr[2], label="Blue", color = 'blue')
    # plt.title("Cross-Correlation")
    # plt.legend()
    # plt.figure()
    # merged_rgb = cv2.merge((dec[0], dec[1], dec[2]))
    # plt.imshow(merged_rgb)
    # plt.title("Merged Flower Channels")
    # plt.axis('off')
    
    plt.show()    
    new_im = cv2.imread("python/test_im/flowerfull.jpg")
    # convert to rgb
    new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
    mse_red = np.mean((new_im[:, :, 0] - dec[0]) ** 2)
    mse_green = np.mean((new_im[:, :, 1] - dec[1]) ** 2)
    mse_blue = np.mean((new_im[:, :, 2] - dec[2]) ** 2)
    
    print(f"MSE in Red Channel: {mse_red}")
    print(f"MSE in Green Channel: {mse_green}")
    print(f"MSE in Blue Channel: {mse_blue}")

    # Plot the differences
    diff_red = np.abs(new_im[:, :, 0] - dec[0])
    diff_green = np.abs(new_im[:, :, 1] - dec[1])
    diff_blue = np.abs(new_im[:, :, 2] - dec[2])
    
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(diff_red, cmap='viridis')
    plt.title("Difference in Red Channel")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(diff_green, cmap='viridis')
    plt.title("Difference in Green Channel")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(diff_blue, cmap= 'viridis')
    plt.title("Difference in Blue Channel")
    plt.axis('off')
    
    plt.show()

def run_estimate_w1_w2_patch(patch, channel, shift_estimation, og_patch = None):
    """
    Run estimation getting w1 and w2
    transformed_image: the image to estimate the weighting for
    """

    w12_vals = []

    print(f"Shift estimate: {shift_estimation}")
    
    img_channel_grey = patch[:, :, channel]
    # save the original patch
    # plt.imsave("patch.png", img_channel_grey, cmap='gray')
    resize_var = 1
    
    # normalise og_patch
    if og_patch is not None:
        og_patch = og_patch[:,:, channel]
        # og_patch = cv2.normalize(og_patch, None, 0, 1, cv2.NORM_MINMAX)
        # og_patch = og_patch / np.max(og_patch)
    # make image larger
    img_channel_grey = cv2.resize(img_channel_grey, (img_channel_grey.shape[1] * resize_var, img_channel_grey.shape[0] * resize_var))
    # histogram equalisation
    shift_estimation = shift_estimation * resize_var
    
    shift, corr = ac.compute_auto_corr(img_channel_grey, shift_estimation, shift_est_func=True, normalised=True)
    shift = np.array(shift)
    corr = np.array(corr)
    cropp_val = 6
    valid_indicies = []
    for i in range(len(shift)):
        if np.abs(shift[i]) <= cropp_val:
            valid_indicies.append(i)
    valid_indicies = np.array(valid_indicies)

    
    if og_patch is not None:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        # Increase contrast using histogram equalization
        # img_channel_grey = cv2.equalizeHist((img_channel_grey * 255).astype(np.uint8)) / 255.0
        im = axs[0, 0].imshow(img_channel_grey[cropp_val:-cropp_val, cropp_val:-cropp_val], cmap='gray', vmin=0, vmax=1)
        axs[0, 0].set_title(f"Channel {channel} Patch Blurred")
        # plt.colorbar(im, ax=axs[0])
        axs[1, 1].plot(shift[valid_indicies], corr[valid_indicies], label = "Blurred")
        axs[1,1].grid('on')
        axs[1, 1].set_title(f"Channel {channel} Patch Cross-Correlation")
        
        fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10))
        im = axs2[0, 0].imshow(img_channel_grey, cmap='gray', vmin=0, vmax=1)
        axs2[0, 0].set_title(f"Channel {channel} Patch Blurred")
        # plt.colorbar(im, ax=axs[0, 1])
        axs2[1, 1].plot(shift, corr, label="Blurred")
        axs2[1, 1].set_title(f"Channel {channel} Patch Blurred Cross-Correlation")
        axs2[1,1].grid('on')
        
        fig3, axs3 = plt.subplots()
        im = axs3.imshow(img_channel_grey, cmap='gray', vmin=0, vmax=1)
        axs3.set_title(f"Channel {channel} Patch Blurred")
        
        # Draw a green box around [-6:6, -6:6]
        rect = plt.Rectangle((img_channel_grey.shape[1]//2 - cropp_val, img_channel_grey.shape[0]//2 - cropp_val), patch.shape[1], patch.shape[0], edgecolor='green', facecolor='none', linewidth=3)
        axs3.add_patch(rect)
        
        fig4, axs4 = plt.subplots()
        axs4.plot(shift, corr, label="Blurred")
        axs4.set_title(f"Channel {channel} Patch Blurred Cross-Correlation", fontsize=22)
        axs4.grid('on')
        axs4.tick_params(axis='both', which='major', labelsize=20)
        
        fig_plots, axs_plots = plt.subplots(1, 3, figsize=(15, 10), constrained_layout=True)
        axs_plots[0].imshow(img_channel_grey[cropp_val:-cropp_val, cropp_val:-cropp_val], cmap='gray', vmin=0, vmax=1)
        axs_plots[0].set_title(f"Channel {channel} Patch Blurred")
        axs_plots[2].plot(shift[valid_indicies], corr[valid_indicies], label="Blurred")
        
        
    
    est, loss, all_vals = WeightingEstimate.optimise_psf_2(img_channel_grey, shift_estimation)
    if og_patch is not None:
        w_vals = [w_ for w_, loss in all_vals]
        loss_vals = [loss for w_, loss in all_vals]
        w1_vals = [w[0] for w in w_vals]
        w2_vals = [w[1] for w in w_vals]
        
        fig_w, axs_w = plt.subplots()
        axs_w.plot(np.arange(1, len(loss_vals) + 1, 1), loss_vals)
        axs_w.set_title("Loss Values", fontsize=22)
        axs_w.grid('on')
        axs_w.tick_params(axis='both', which='major', labelsize=20)
        
        fig_w2, axs_w2 = plt.subplots()
        axs_w2.plot(np.arange(1, len(w1_vals) + 1, 1), w1_vals, label="w1")
        axs_w2.plot(np.arange(1, len(w2_vals) + 1, 1), w2_vals, label="w2")
        axs_w2.set_title("W1 and W2 Values", fontsize=22)
        axs_w2.grid('on')
        axs_w2.legend(fontsize=20)
        axs_w2.tick_params(axis='both', which='major', labelsize=20)
    
    w12_vals = [est[0], est[1]]
    

    # # deconvolved = WeightingEstimate.deconvolve_img(img_channel_grey, WeightingEstimate.get_img_psf(est1, shift_estimation))
    deconvolved = WeightingEstimate.deconvolve_img(img_channel_grey, WeightingEstimate.get_img_psf_2(w12_vals[0], w12_vals[1], shift_estimation))
    
    if og_patch is not None:
        shift_dec, corr_dec = ac.compute_auto_corr(deconvolved, shift_estimation, shift_est_func=True, normalised=True)
        shift_dec = np.array(shift_dec)
        corr_dec = np.array(corr_dec)
        corr_diffs = []
        im = axs[0, 1].imshow(og_patch[cropp_val:-cropp_val, cropp_val:-cropp_val], cmap='gray', vmin=0, vmax=1)
        axs[0, 1].set_title(f"Channel {channel} Patch Original")
        plt.colorbar(im, ax=axs[0, 1])
        shift_og, corr_og = ac.compute_auto_corr(og_patch, shift_estimation, shift_est_func=True, normalised=True)
        shift_og = np.array(shift_og)
        corr_og = np.array(corr_og)
        axs[1, 1].plot(shift_og[valid_indicies], corr_og[valid_indicies], label="Original")
        axs[1, 1].set_title(f"Channel {channel} Patch Original Cross-Correlation")
        # Increase contrast using histogram equalization for the deconvolved image
        im = axs[1, 0].imshow(deconvolved[cropp_val:-cropp_val, cropp_val:-cropp_val], cmap='gray', vmin=0, vmax=1)
        axs[1, 0].set_title(f"Channel {channel} Patch Deconvolved")
        # plt.colorbar(im, ax=axs[1, 0])
        axs[1, 1].plot(shift_dec[valid_indicies], corr_dec[valid_indicies], label="Deconvolved")
        axs[1,1].legend()
        
        
        
        im = axs2[0,1].imshow(og_patch, cmap='gray', vmin=0, vmax=1)
        axs2[0, 1].set_title(f"Channel {channel} Patch Original")
        
        axs2[1, 1].plot(shift_og, corr_og, label="Original")
        axs2[1, 1].set_title(f"Channel {channel} Patch Original Cross-Correlation")
        
        im = axs2[1, 0].imshow(deconvolved, cmap='gray', vmin=0, vmax=1)
        axs2[1, 0].set_title(f"Channel {channel} Patch Deconvolved")
        axs2[1, 1].plot(shift_dec, corr_dec, label="Deconvolved")
        axs2[1,1].legend()
        axs_plots[0].imshow(deconvolved[cropp_val:-cropp_val, cropp_val:-cropp_val], cmap='gray', vmin=0, vmax=1)
        # axs_plots[1].imshow(deconvolved_1[cropp_val:-cropp_val, cropp_val:-cropp_val], cmap='gray', vmin=0, vmax=1)
        shift_dec1, corr_dec1 = ac.compute_auto_corr(deconvolved, shift_estimation, shift_est_func=True, normalised=True)
        shift_dec1 = np.array(shift_dec1)
        corr_dec1 = np.array(corr_dec1)
        axs_plots[2].plot(shift_dec1[valid_indicies], corr_dec1[valid_indicies], label="Deconvolved N = 0.01")
        axs_plots[0].set_title(f"Channel {channel} Patch Deconvolved N = 0.01")
        axs_plots[1].set_title(f"Channel {channel} Patch Deconvolved N = 0.1")
        axs_plots[2].plot(shift_dec[valid_indicies], corr_dec[valid_indicies], label="Deconvolved N = 0.1")
        axs_plots[2].plot(shift_og[valid_indicies], corr_og[valid_indicies], label="Original")
        axs_plots[2].set_title("Normalised Cross Correlation")
        axs_plots[2].legend()
        for ax in axs_plots:
            ax.set_box_aspect(1)
        
        # # stack the deconvolved images
        plt.tight_layout()
        # plt.figure()
        # convolved_blur = sp.signal.fftconvolve(deconvolved, WeightingEstimate.get_img_psf_2(w12_vals[0], w12_vals[1], shift_estimation), mode='same')
        # plt.imshow(convolved_blur, cmap='gray', vmin=0, vmax=1)
        # plt.title("Deconvolved Image Convolved with PSF")
        plt.show()
    
    print(w12_vals)
    
    
    # est_0, _, all_vals_0 = WeightingEstimate.optimise_psf_2(img_channel_grey, shift_estimation, balance=0)
    # w12_vals_0 = [est_0[0], est_0[1]]
    # print(f"w12 vals 0 {w12_vals_0}")
    # deconvolved_0 = WeightingEstimate.deconvolve_img(img_channel_grey, WeightingEstimate.get_img_psf_2(w12_vals_0[0], w12_vals_0[1], shift_estimation), balance=0)
    # shift_0, corr_0 = ac.compute_auto_corr(deconvolved_0, shift_estimation, shift_est_func=True, normalised=True)
    # shift_0 = np.array(shift_0)
    # corr_0 = np.array(corr_0)
    
    # est_01, _, all_vals_01 = WeightingEstimate.optimise_psf_2(img_channel_grey, shift_estimation, balance=0.1)
    # w12_vals_01 = [est_01[0], est_01[1]]
    # deconvolved_01 = WeightingEstimate.deconvolve_img(img_channel_grey, WeightingEstimate.get_img_psf_2(w12_vals_01[0], w12_vals_01[1], shift_estimation), balance=0.1)
    # shift_01, corr_01 = ac.compute_auto_corr(deconvolved_01, shift_estimation, shift_est_func=True, normalised=True)
    # shift_01 = np.array(shift_01)
    # corr_01 = np.array(corr_01)
    # print(f"w12 vals 01 {w12_vals_01}")
    # est_001, _, all_vals_001 = WeightingEstimate.optimise_psf_2(img_channel_grey, shift_estimation, balance=0.01)
    # w12_vals_001 = [est_001[0], est_001[1]]
    # deconvolved_001 = WeightingEstimate.deconvolve_img(img_channel_grey, WeightingEstimate.get_img_psf_2(w12_vals_001[0], w12_vals_001[1], shift_estimation), balance=0.01)
    # shift_001, corr_001 = ac.compute_auto_corr(deconvolved_001, shift_estimation, shift_est_func=True, normalised=True)
    # shift_001 = np.array(shift_001)
    # corr_001 = np.array(corr_001)
    # print(f"w12 vals 001 {w12_vals_001}")
    # fig, axs = plt.subplots(2,2 , figsize=(15, 10))
    # axs[0, 0].imshow(deconvolved_0[cropp_val:-cropp_val, cropp_val:-cropp_val], cmap='gray', vmin=0, vmax=1)
    # axs[0, 0].set_title("Deconvolved i = 2", fontsize=20)
    # axs[0, 0].axis('off')
    
    # axs[0,1].imshow(deconvolved_01[cropp_val:-cropp_val, cropp_val:-cropp_val], cmap='gray', vmin=0, vmax=1)
    # axs[0,1].set_title("Deconvolved i = 3", fontsize=20)
    # axs[0,1].axis('off')
    
    # axs[1,0].imshow(deconvolved_001[cropp_val:-cropp_val, cropp_val:-cropp_val], cmap='gray', vmin=0, vmax=1)
    # axs[1,0].set_title("Deconvolved i = 4", fontsize=20)
    # axs[1,0].axis('off')
    
    
    # axs[1,1].plot(shift_0[valid_indicies], corr_0[valid_indicies], label="Deconvolved i = 2")
    # axs[1,1].plot(shift_01[valid_indicies], corr_01[valid_indicies], label="Deconvolved i = 3")
    # axs[1,1].plot(shift_001[valid_indicies], corr_001[valid_indicies], label="Deconvolved i = 4")
    # axs[1,1].plot(shift_og[valid_indicies], corr_og[valid_indicies], label="Original")
    # axs[1,1].plot(shift[valid_indicies], corr[valid_indicies], label="Blurred")
    # axs[1,1].grid('on')
    # # axs[1,1].set_title("Normalised Cross Correlation")
    # axs[1,1].legend()
    # for ax in axs.flat:
    #     ax.tick_params(axis='both', which='major', labelsize=20)
    # # icnrease legend size
    # axs[1,1].legend(fontsize=14)
    # # title size
    # axs[1,1].set_title("Normalised Cross Correlation", fontsize=20)
    # plt.tight_layout()
    # plt.show()
    
    # for ax in axs:
    #         ax.set_box_aspect(1)
    return deconvolved, w12_vals, None

# def run_estimate_w1_w2_patch(patch, channel, shift_estimation, og_patch=None, neighbor_weights=None):
#     """
#     Estimate `w1` and `w2` for each patch using localized neighbor weights.
#     """
#     img_channel_grey = patch[:, :, channel]
#     # shift, corr = ac.compute_auto_corr(img_channel_grey, shift_estimation, shift_est_func=True, normalised=True)

#     # Run the weighted PSF optimization with optional neighbor weights for regularization
#     est_w1, loss, all_vals, _ = WeightingEstimate.optimise_psf(
#         img_channel_grey, shift_estimation, neighbor_weights=neighbor_weights)
#     est_w2 = 1 - est_w1

#     w12_vals = [est_w1, est_w2]
#     deconvolved = WeightingEstimate.deconvolve_img(img_channel_grey, WeightingEstimate.get_img_psf(est_w1, shift_estimation))
#     print(w12_vals, loss)
#     return deconvolved, w12_vals, None