#!/.bashrc
# NOTE: MUST BE RUNNED IN extrinsics_visualization directory
conda activate nstudio

WORKDIR=/scratch-ssd/dense_workdir
# COLMAP_DIR=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/Videos/book_sofa_recons
COLMAP_DIR=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/sofa_soccer_dragon

WORK_COLMAP=$WORKDIR/$(basename $COLMAP_DIR)

mkdir $WORKDIR
echo copying colmap to $WORK_COLMAP
cp -r $COLMAP_DIR $WORKDIR
python create_del_list.py --img_dir $WORK_COLMAP/images


echo deleting images
cp $WORK_COLMAP/database.db $WORK_COLMAP/sparse/0
mkdir -p $WORK_COLMAP/new_sparse/0
colmap image_deleter --input_path $WORK_COLMAP/sparse/0 \
                     --output_path $WORK_COLMAP/new_sparse/0 \
                     --image_names_path $WORK_COLMAP/del_paths.txt


mkdir $WORK_COLMAP/dense


## dense reconstruction
colmap image_undistorter \
  --image_path $WORK_COLMAP/images \
  --input_path $WORK_COLMAP/new_sparse/0 \
  --output_path $WORK_COLMAP/dense \
  --output_type COLMAP

colmap patch_match_stereo \
  --workspace_path $WORK_COLMAP/dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.window_radius 9 \
  --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.filter_min_ncc .07

colmap stereo_fusion \
  --workspace_path $WORK_COLMAP/dense \
  --workspace_format COLMAP \
  --input_type photometric \
  --output_path $WORK_COLMAP/dense/fused.ply


## output bin to txt

mkdir $WORK_COLMAP/dense/txt_sparse
echo $WORK_COLMAP/dense/sparse
colmap model_converter \
    --input_path $WORK_COLMAP/dense/sparse \
    --output_path $WORK_COLMAP/dense/txt_sparse \
    --output_type TXT

# ffmpeg -framerate 24 -i vid_frames/%04d.png -c:v libx264 -pix_fmt yuv420p video.mp4
# ffmpeg -framerate 24 -i vid_frames/%04d.png -c:v libx264 -pix_fmt yuv420p -frames:v 5000 video.mp4
# ffmpeg -framerate 30 -i sofa_soccer_dragon_vid_frames/%04d.png -c:v libx264 -pix_fmt yuv420p -frames:v 5000 sofa_soccer_dragon_video.mp4
