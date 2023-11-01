DATASET_PATH=timer_rgb_checker_recons

echo feature extracting ...
colmap feature_extractor \
 --database_path $DATASET_PATH/database.db \
 --image_path $DATASET_PATH/images \
 --ImageReader.camera_model OPENCV \
 --ImageReader.single_camera 1

echo matching ...

colmap sequential_matcher \
 --database_path $DATASET_PATH/database.db \


mkdir $DATASET_PATH/sparse

echo sparse reconstruction ...

colmap mapper \
  --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH/images \
  --output_path $DATASET_PATH/sparse

# mkdir $DATASET_PATH/dense

# colmap image_undistorter \
#     --image_path $DATASET_PATH/images \
#     --input_path $DATASET_PATH/sparse/0 \
#     --output_path $DATASET_PATH/dense \
#     --output_type COLMAP \
#     --max_image_size 2000

