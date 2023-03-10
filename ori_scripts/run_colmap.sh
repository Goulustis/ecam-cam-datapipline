## run colmap to get extrinsics
## assume images are in {scene}_recon/images

DATASET_PATH=checker_recon

echo feature extracting ...
colmap feature_extractor \
 --database_path $DATASET_PATH/database.db \
 --image_path $DATASET_PATH/images \
 --ImageReader.camera_model OPENCV \
 --ImageReader.single_camera 1

echo matching ...

colmap sequential_matcher \
 --database_path $DATASET_PATH/database.db


mkdir $DATASET_PATH/sparse

echo sparse reconstruction ...

colmap mapper \
  --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH/images \
  --output_path $DATASET_PATH/sparse

