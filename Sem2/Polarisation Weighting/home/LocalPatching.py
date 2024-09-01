import WeightingEstimate
import Autocorrelation as ac
import numpy as np
from skimage import restoration as sk
import scipy as sp
### Per Pixel Local Patching ###
### The following is for a single patch per pixel at a time
### ###
def get_local_patch(img, x, y, patch_size = 50):
    """
    Extract a local patch around the pixel (x, y) in the image
    img: the image to extract the patch from
    x: the x coordinate of the pixel
    y: the y coordinate of the pixel
    patch_size: the size of the patch to extract
    """
    half_patch = patch_size // 2
    patch = img[max(0, y-half_patch):min(y+half_patch+1, img.shape[0]), 
                max(0, x-half_patch):min(x+half_patch+1, img.shape[1])]
    return patch

def loss_patch(w, patch, shift_val, loss_vals, w_vals):
    # w1_loc, w2_loc = w


    if np.max(patch) > 1:
        print("Normalising as max value has been reached in loss function two set")
        patch = patch / np.max(patch)

    w1_loc = w
    psf_estimate = WeightingEstimate.get_img_psf(w1_loc, shift_val)
    # psf_estimate = WeightingEstimate.get_img_psf_w1_w2(w1_loc, w2_loc, shift_val)

    deconvolved_patch = sk.wiener(patch, psf_estimate, balance=0)

    deconvolved_patch = deconvolved_patch/np.max(deconvolved_patch)

    # obtain tyhe correlation values of the deconvolved image
    shift_vals, corr_vals = ac.compute_auto_corr(deconvolved_patch, shift_val, shift_est_func=False, normalised=True)
    # apply a savgol filter to the correlation values for peak detection
    corr_filt = ac.obtain_peak_highlighted_curve(corr_vals)

    # develop loss
    loss = 0
    # loss = check_gradients(corr_filt, shift_vals)
    loss = WeightingEstimate.check_gradients(corr_vals, shift_vals)
    loss += WeightingEstimate.check_flatness(shift_vals, corr_filt, shift_val)
    loss += WeightingEstimate.minimise_corr_vals(corr_filt, shift_vals)

    loss_vals.append(loss)
    w_vals.append(w)

    min_val = np.where(loss_vals == np.min(loss_vals))[0][0]

    # print(f"Loss: {loss.min()} and w1: {w1_vals[min_val]}")

    return loss

def estimate_local_weights(patch, shift_val):
    """
    Estimate the weights for the local patch
    patch: the patch to estimate the weights for    
    shift_val: the shift value for the patch
    """

    loss_vals = []
    w_vals = []
    
    # differential evolu8tion optimisation
    # BOUNDS = [(0, 1), (0, 1)]
    BOUNDS = [(0,1)]

    result = sp.optimize.differential_evolution(loss_patch, 
                                                bounds=BOUNDS, 
                                                args=(patch, shift_val, loss_vals, w_vals),
                                                disp=True,
                                                polish=False, # use L-BFGS-B to polish the best result
                                                workers=-1)
    
    w1_estimate = result.x[0]
    # w2_estimate = result.x[1]
    loss_value = result.fun

    # est_w1, est_w2 = WeightingEstimate.clarity_both_imgs_w1w2(patch, w1_estimate, w2_estimate, shift_val)

    # return est_w1, est_w2
    return w1_estimate, 1-w1_estimate

def apply_local_deconvolution(img, shift_val, patch_size=50):
    """
    Apply local deconvolution to the image
    img: the image to apply the deconvolution to
    shift_val: the shift value to apply to the image
    patch_size: the size of the patch to apply the deconvolution to
    """

    deconvolved_img = np.zeros_like(img)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            patch = get_local_patch(img, x, y, patch_size)
            w1, w2 = estimate_local_weights(patch, shift_val)
            print(f"Estimated w1: {w1} and w2: {w2}")
            psf_estimate = WeightingEstimate.get_img_psf_w1_w2(w1, w2, shift_val)
            deconvolved_patch = sk.wiener(patch, psf_estimate, balance=0)
            deconvolved_patch = deconvolved_patch / np.max(deconvolved_patch)
            deconvolved_img[y, x] = deconvolved_patch[patch_size//2, patch_size//2]

    return deconvolved_img

### Local Patching ###
### The following is for a finite number of patches
### ###

def get_image_patches(img, num_patches_x, num_patches_y):
    """
    Divide the image into a grid of patches.
    img: The input image to be divided.
    num_patches_x: Number of patches along the x-axis.
    num_patches_y: Number of patches along the y-axis.
    """
    patch_size_x = img.shape[1] // num_patches_x
    patch_size_y = img.shape[0] // num_patches_y
    
    patches = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            patch = img[i*patch_size_y:(i+1)*patch_size_y, j*patch_size_x:(j+1)*patch_size_x]
            patches.append(patch)
    return patches, patch_size_x, patch_size_y

def apply_deconvolution_to_patches(img, shift_val, num_patches_x=5, num_patches_y=5):
    """
    Apply local deconvolution to the image using a grid of patches.
    img: The input image to be deconvolved.
    shift_val: The shift value for the PSF.
    num_patches_x: Number of patches along the x-axis.
    num_patches_y: Number of patches along the y-axis.
    """
    patches, patch_size_x, patch_size_y = get_image_patches(img, num_patches_x, num_patches_y)
    
    deconvolved_img = np.zeros_like(img)
    
    for i, patch in enumerate(patches):
        w1, w2 = estimate_local_weights(patch, shift_val)
        print(f"Estimated w1: {w1} and w2: {w2} for patch {i+1}/{len(patches)}")
        psf_estimate = WeightingEstimate.get_img_psf_w1_w2(w1, w2, shift_val)
        deconvolved_patch = sk.wiener(patch, psf_estimate, balance=0)
        deconvolved_patch = deconvolved_patch / np.max(deconvolved_patch)
        
        # Determine where to place the deconvolved patch in the final image
        row_start = (i // num_patches_x) * patch_size_y
        col_start = (i % num_patches_x) * patch_size_x
        deconvolved_img[row_start:row_start+patch_size_y, col_start:col_start+patch_size_x] = deconvolved_patch
    
    return deconvolved_img