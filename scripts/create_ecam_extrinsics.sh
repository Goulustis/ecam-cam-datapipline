# NOTE: run find_relative_cam.sh to find rel_cam.json
# This script will find the event camera extrinsics 

REL_CAM_PATH=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/boardroom_b2_v1/rel_cam.json                         # [REQUIRED] relative camera positions
SCENE_PATH=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/boardroom_b2_v1                                      # [REQUIRED] path to your scene (eg. some_path/scene)
SCENE=$(basename $SCENE_PATH)
PNT_3D_PATH=$SCENE_PATH/${SCENE}_recon/sparse/0/points3D.bin   # output of colmap
IMG_PATH=$SCENE_PATH/${SCENE}_recon/sparse/0/images.bin        # output of colmap
SCALE_PNT_PATH=$SCENE_PATH/${SCENE}_recon/scale_pnts.txt       # path containing id of 3d points
REAL_SCALE_PATH=$SCENE_PATH/${SCENE}_recon/pnt_dist.txt        # path containing actual distance of 3d points in (cm, mm, ...) is 
                                                               #      consistent with unit of square for finding relative camera params

COLMAP_SCALE_PATH=$SCENE_PATH/${SCENE}_recon/colmap_scale.txt

## find real scale of colmap
## save scale under $SCENE_PATH/$SCENE/colmap_scale.txt
echo finding the scale
python colmap_find_scale/find_scale.py -c $PNT_3D_PATH -p $SCALE_PNT_PATH -l $REAL_SCALE_PATH

# CREATE THE EXTRININSICS
# 1) colmap quaterion to extrinsic matrix
# default saved to $(dirname $IMG_PATH)/images_mtx.npy
echo turning colmap quaterion to extrinsics
COL_EXT_CAM_PATH=$(dirname $IMG_PATH)/images_mtx.npy

# cd extrinsics_creator
python extrinsics_creator/qua_to_ext.py -i $IMG_PATH -o $COL_EXT_CAM_PATH

# CREATE THE RELATIVE CAMERA
# save result under $SCENE_PATH/e_cams.npy
echo creating event camera extrinsics
python extrinsics_creator/create_rel_cam.py -r $REL_CAM_PATH \
                                            -c $COL_EXT_CAM_PATH \
                                            -s $COLMAP_SCALE_PATH
