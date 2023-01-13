# This will turn color images into a formatted dataset

IMG_DIR=data/checker/original_images            # original images
COLMAP_DIR=data/checker/checker_recon/sparse/0      # output of colmap
IMG_SCALE=2                                         # scale of image used for colmap
TARG_DIR=data/formatted_checkers                    # location to save formatted dataset
TRIGGER_PATH=data/checker/triggers.txt
T_SCALE_FACTOR=1

python format_data/format_col_set.py --img_dir $IMG_DIR \
                                     --colmap_dir $COLMAP_DIR \
                                     --img_scale $IMG_SCALE \
                                     --target_dir $TARG_DIR \
                                     --blurry_per 0 \
                                     --trigger_path $TRIGGER_PATH \
                                     --t_scale_factor $T_SCALE_FACTOR

cp -r $IMG_DIR $TARG_DIR
cd $TARG_DIR
mv $(basename $IMG_DIR) 1x
mkdir rgb
mv 1x rgb