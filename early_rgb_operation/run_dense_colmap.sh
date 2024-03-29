DATASET_PATH=rgb_checker_recons

mkdir $DATASET_PATH/dense

#colmap image_undistorter \
#  --image_path $DATASET_PATH/images \
#  --input_path $DATASET_PATH/sparse/0 \
#  --output_path $DATASET_PATH/dense \
#  --output_type COLMAP \
#  --max_image_size 2000

colmap patch_match_stereo \
  --workspace_path $DATASET_PATH/dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
  --workspace_path $DATASET_PATH/dense \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path $DATASET_PATH/dense/fused.ply

colmap poisson_mesher \
  --input_path $DATASET_PATH/dense/fused.ply \
  --output_path $DATASET_PATH/dense/meshed-poisson.ply
