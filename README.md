# Event Camera dataset pipeline

this repository will turn images and events into a dataset scene


## installation
1) install packages in requirement.txt
2) install [e2calib](https://github.com/uzh-rpg/e2calib) in the third party directory 
3) For visualization, install [this](https://github.com/Goulustis/exviz) and open3d

## Instructions
The scripts should be runned in the following sequence:
1) scripts/find_relative_cam.sh      (finds relative camera position; checker board scene required)
2) scripts/create_ecam_extrinsics.sh (finds colmap scene scale & event camera extrinsics based on relative camera position)
3) scripts/format_dataset.sh         (creates one scene)

## input directory format
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
|   |   scale_pnts.txt               (points in colmap selected for finding scene scale)
|   |   pnt_dist.txt                 (actual length of selected points in real world)
│   events.h5                        (the event data of the scene; in metavision v3.0)
|   triggers.txt                     (triggers received; ON events only; 1 events only)
|   rel_cam.json                     (color to event camera R,t; obtained via find_relative_cam.sh)
|   e_cams.npy                       (corresponding event camera extrinsics at trigger times; obtained via create_ecam_extrinsics.sh)
```

Inside scale_pnts.txt will have 2 idxs:
```
idx_1
idx_2 
```

Inside pnt_dist points will the real distance of between the two points in idx_1, idx_2:
```
4.32
```

## output directory
```
<dataset_name>
|   rel_cam.json ([not used] check stereo_calib/camera_calibrate.py for details)
└───colcam_set   (mostly the same as nerfies)
|   |   dataset.json
|   |   metadata.json
|   |   scene.json
|   └───camera   (same as in nerfies)
|   |   |    000000.json
|   |   |    ...
|   └───rgb
|       └───1x  (original image size)
|       |    |   00000.png
|       |    |   ....
|       └───2x  (down sized 2x image)
|            |   ....    
└───ecam_set
|   |   dataset.json
|   |   metadata.json
|   |   scene.json
|   |   prcessed_event.h5  (event compatible with prosphesees v2)
|   └───camera             (same as above)
|   └───eimgs
|       |   eimgs_1x.npy   (the accumulated events frames)
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
