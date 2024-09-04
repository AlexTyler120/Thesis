import ImageRun
import Segmentation
import matplotlib.pyplot as plt
def run_patch_estimation(img):

    patches = Segmentation.segment_image_patches(img)
    new_patches = []
    for patch in patches:
        img_patch = ImageRun.run_estimate_w1_w2(patch)
        new_patches.append(img_patch)

    recombined_image = Segmentation.recombine_patches(new_patches, img.shape[0], img.shape[1])
    plt.figure()
    plt.imshow(recombined_image)
    plt.show()

