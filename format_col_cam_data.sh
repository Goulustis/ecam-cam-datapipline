IMG_DIR=/home/matthew/Videos/checker_png
COLMAP_DIR=data/checker/checker_recon/sparse/0
IMG_SCALE=2
TARG_DIR=data/formated_checkers

python format_data/format_col_set.py --img_dir $IMG_DIR --colmap_dir $COLMAP_DIR --img_scale $IMG_SCALE --target_dir $TARG_DIR