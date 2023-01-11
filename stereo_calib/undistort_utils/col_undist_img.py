import numpy as np
import cv2
from tqdm import tqdm
from e_img_undistort_v2 import intr_mtx, dist

te_img_f = "/scratch/matthew/projects/event_calib/stereo_calibration/data/checker_RIGHT/000159.JPG"

# intr_mtx = np.array([
#         [
#             1093.9944355005182,
#             0.0,
#             750.4162607164546
#         ],
#         [
#             0.0,
#             1093.661550555695,
#             577.865032906051
#         ],
#         [
#             0.0,
#             0.0,
#             1.0
#         ]
#     ])

# dist = np.array( [
#             -0.39951800872807586,
#             0.30129385849163637,
#             -0.0004109200082123863,
#             0.0008991939667037576,
#             -0.18783047606008396
#         ])

def my_remap(mapx, mapy, src_img, target_size):
    # target_size = (h ,w)
    h, w = target_size
    img = np.zeros((h, w))

    valid_coords = ((0 <= mapx) & (mapx < w)) & \
                    ((0 <= mapy) & (mapy < h))  
    
    xx,yy = np.linspace(w), np.linspace(h)
    xx, yy = np.meshgrid(xx, yy)
    xx, yy = xx[valid_coords], yy[valid_coords]

    for dst_x,dst_y in zip(xx, yy): #zip(xx.reshape(-1), yy.reshape(-1)):
        src_x, src_y = mapx[dst_x], mapy[dst_y]
        img[dst_y, dst_x] = src_img[src_y, src_x]
    
    return img
    



def main():
    # img_f = "/scratch/matthew/projects/event_calib/stereo_calibration/data/checker_LEFT/000159.JPG"
    img_f = te_img_f

    img = cv2.imread(img_f, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    scale = 1
    w1, h1 = int(scale*w),int(scale*h)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intr_mtx, dist, (w,h), 1, (w1,h1))
    
    mapx, mapy = cv2.initUndistortRectifyMap(intr_mtx, dist, None, newcameramtx, (w1,h1), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # dst = my_remap(mapx, mapy, img, (h1,w1))

    x,y,w,h = roi 
    cv2.imwrite('inter_calibresult.png', dst)
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)
    print("new_res", dst.shape)

    # old
    # new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(intr_mtx, dist, (w,h), 1, (w,h))
    # dst = cv2.undistort(img, intr_mtx, dist, None, new_cam_mtx)


if __name__ == "__main__":
    main()