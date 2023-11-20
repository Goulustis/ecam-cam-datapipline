source ~/.bashrc
PREP_EVENT_F=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/halloween_b2_v1/processed_events.h5
# EVS_RECON_IMG_DIR=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/atrium_b2_v1/ev_imgs
# EVS_IMG_TRIG_DIR=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/atrium_b2_v1/trig_eimgs
# COLMAP_DIR=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/atrium_b2_v1/atrium_b2_v1_recon
# RELCAM_F=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/atrium_b2_v1/rel_cam.json

EVS_RECON_IMG_DIR=$(dirname $PREP_EVENT_F)/ev_imgs
EVS_IMG_TRIG_DIR=$(dirname $PREP_EVENT_F)/trig_eimgs
COLMAP_DIR=$(dirname $PREP_EVENT_F)/$(basename $(dirname $PREP_EVENT_F))_recon
RELCAM_F=$(dirname $PREP_EVENT_F)/rel_cam.json
TRIGGER_F=$(dirname $PREP_EVENT_F)/triggers.txt
ECAM_F=$(dirname $PREP_EVENT_F)/e_cams.npy
# if [ ! -e "$EVS_RECON_IMG_DIR" ]; then
#     WORKDIR=$(pwd)
#     conda activate e2calib 
#     cd third_party/e2calib/python
#     echo generating event images ...
#     python offline_reconstruction.py --h5file $PREP_EVENT_F \
#                                      --timestamps_file $TRIGGER_F \
#                                      --output_folder $EVS_RECON_IMG_DIR
#     conda deactivate
#     cd $WORKDIR
# fi


conda activate ecam_proc
if [ ! -e "$EVS_IMG_TRIG_DIR" ]; then
    python extrinsics_visualization/make_trig_eimgs.py -i $PREP_EVENT_F \
                                                    -t $TRIGGER_F \
                                                    -o $EVS_IMG_TRIG_DIR
fi

python extrinsics_visualization/viz_sparse_reporjection.py --colmap_dir $COLMAP_DIR \
                                                   --ev_imgs_dir $EVS_IMG_TRIG_DIR \
                                                   --ecam_f $ECAM_F \
                                                   --relcam_f $RELCAM_F
