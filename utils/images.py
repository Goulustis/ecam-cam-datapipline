import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import imageio
from tqdm import tqdm
from utils.misc import parallel_map


def calc_clearness_score(img_list, ignore_first = 0):
    """
    inp:
        img_list (list [str]): path to image files
    output:
        clear_img_fs (list [str]): sorted images files from clearest to blurriest
        best (list [int]): sorted idx of original img_list from clearest to blurriest
        blur_scores (list [float]): blur score of each images (higher is clearer)

    """
    # Get list of images in folder
    img_list = img_list[ignore_first:]

    # Load images
    images = parallel_map(imageio.imread, img_list, show_pbar=True, desc="loading imgs")

    blur_scores = []
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)
    blur_kernels = np.array([[
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ], [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], [
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ]], dtype=np.float32) / 5.0
    
    def calc_blur(image):
        gray_im = np.mean(image, axis=2)[::4, ::4]

        directional_blur_scores = []
        for i in range(4):
            blurred = ndimage.convolve(gray_im, blur_kernels[i])

            laplacian = signal.convolve2d(blurred, laplacian_kernel, mode="valid")
            var = laplacian**2
            var = np.clip(var, 0, 1000.0)

            directional_blur_scores.append(np.mean(var))

        antiblur_index = (np.argmax(directional_blur_scores) + 2) % 4

        return directional_blur_scores[antiblur_index]

    blur_scores = parallel_map(calc_blur, images, show_pbar=True, desc="calculating blur score")
    
    ids = np.argsort(blur_scores) + ignore_first
    best = ids[::-1]
 
    clear_image_fs = [img_list[e] for e in best]
    return clear_image_fs, best, np.array(blur_scores)