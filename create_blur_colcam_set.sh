#!/bin/bash
conda activate ecam_proc

################# INPUTS ############
COLCAM_SET_DIR=/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/calib_checker/colcam_set
NUM_BLUR=6

python format_data/format_blur_colcam_set.py --src_dir $COLCAM_SET_DIR \
                                             --num_blur $NUM_BLUR