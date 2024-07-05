### choose scene here; make the workdir first
source ~/.bashrc
conda activate ecam_proc

SCENE=lab_c1
N_BINS=4
DELTA_T=5000  # in microsce; None for turning it off

WORKING_DIR=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/${SCENE}
TARG_DIR=/ubc/cs/research/kmyi/matthew/projects/E2NeRF/data/real-world/${SCENE}
EVENT_H5=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/$SCENE/events.h5

TARGET_EVENT_H5=$WORKING_DIR/processed_events.h5


# CREATE EVENT IMAGES
if [ ! -e "$TARGET_EVENT_H5" ]; then
    echo formatting event file for e2calib
    echo $TARGET_EVENT_H5
    python data_utils/process_event_format.py -i $EVENT_H5 -o $TARGET_EVENT_H5 -t $TRIGGER_PATH
fi

python format_data/format_e2nerf.py --work_dir ${WORKING_DIR} \
                                    --targ_dir ${TARG_DIR} \
                                    --n_bins ${N_BINS} \
                                    --delta_t $DELTA_T