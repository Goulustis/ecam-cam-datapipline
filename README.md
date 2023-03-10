# Event Camera dataset pipeline

this repository will turn images and events into a dataset scene


## installation
1) install packages in requirement.txt
2) install [e2calib](https://github.com/uzh-rpg/e2calib) in the third party directory 
3) For visualization, install [this](https://github.com/Goulustis/exviz) and open3d

## Instructions
The scripts should be runned in the following sequence:
1) find_relative_cam.sh      (finds relative camera position; checker board scene required)
2) create_ecam_extrinsics.sh (finds colmap scene scale & event camera extrinsics based on relative camera position)
3) format_dataset.sh         (creates one scene)

## directory format
Assume scene structure in this format (both general scene and checker board scene for calibration)
```
scene
└───scene_recon (from colmap reconstruction)
|   |   images
|   |   |   00001.{JPG, PNG, ...}
|   |   |   00002.{JPG, PNG, ...}
|   |   |   ....
|   |   sparse
|   |   └───0
|   |   |   |   images.bin
|   |   |   |   points3D.bin
|   |   |   |   cameras.bin
|   |   scale_pnts.txt               (points selected for finding scene scale)
|   |   pnt_dist.txt                 (actual length of selected points in real world)
│   events.h5                        (the event data of the scene; in metavision v3.0)
|   triggers.txt                     (triggers received; on events only; 1 events only)
|   rel_cam.json                     (color to event camera R,t; obtained via find_relative_cam.sh)
|   e_cams.npy                       (corresponding event camera extrinsics at trigger times; obtained via create_ecam_extrinsics.sh)
```

Inside scale points will have 2 idxs:
```
idx_1
idx_2 
```

Inside pnt_dist points will the real distance of between the two points in idx_1, idx_2:
```
4.32
```

# Note:
For mean_t, switch trigger.txt to mean ts

# Code Acknowledgement
```
https://github.com/uzh-rpg/e2calib
https://github.com/bvnayak/stereo_calibration
https://github.com/google/nerfies
https://github.com/apple/ml-neuman/
```
