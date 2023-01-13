# This will turn color images into a formatted dataset

IMG_DIR=/home/matthew/Videos/checker_png            # original images
COLMAP_DIR=data/checker/checker_recon/sparse/0      # output of colmap
IMG_SCALE=2                                         # scale of image used for colmap
TARG_DIR=data/formatted_checkers                    # location to save formatted dataset

python format_data/format_col_set.py --img_dir $IMG_DIR \
                                     --colmap_dir $COLMAP_DIR \
                                     --img_scale $IMG_SCALE \
                                     --target_dir $TARG_DIR \
                                     --blurry_per 0

cp -r $IMG_DIR $TARG_DIR
cd $TARG_DIR
mv $(basename $IMG_DIR) 1x
mkdir rgb
mv 1x rgb