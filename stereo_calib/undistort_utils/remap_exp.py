# from __future__ import print_function
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# import argparse
def update_map(ind, map_x, map_y):
    if ind == 0:
        for i in range(map_x.shape[0]):
            for j in range(map_x.shape[1]):
                if j > map_x.shape[1]*0.25 and j < map_x.shape[1]*0.75 and i > map_x.shape[0]*0.25 and i < map_x.shape[0]*0.75:
                    map_x[i,j] = 2 * (j-map_x.shape[1]*0.25) + 0.5
                    map_y[i,j] = 2 * (i-map_y.shape[0]*0.25) + 0.5
                else:
                    map_x[i,j] = 0
                    map_y[i,j] = 0
    elif ind == 1:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]
    elif ind == 2:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [map_x.shape[1]-x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [y for y in range(map_y.shape[0])]
    elif ind == 3:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [map_x.shape[1]-x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]


src = cv.imread(cv.samples.findFile("calibresult.png"), cv.IMREAD_COLOR)

map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
window_name = 'Remap demo'

ind = 0


update_map(ind, map_x, map_y)

# dst = cv.remap(src, map_x, map_y, cv.INTER_LINEAR)
# plt.imshow(dst)

coords = []
for x, y in zip(map_x.reshape(-1), map_y.reshape(-1)):
    coords.append(",".join([str(x), str(y)]))

coords = np.array(coords)
uniq, cnt = np.unique(coords, return_counts=True)

for el, c in zip(uniq, cnt):
    if c > 1:
        print(el)

