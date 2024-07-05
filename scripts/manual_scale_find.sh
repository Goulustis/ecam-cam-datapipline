source ~/.bashrc

conda activate evflow 

## INPUT
SCENE=lab_c1

#=====================================================
EVENT_H5=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/$SCENE/events.h5
EV_F=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/$SCENE/processed_events.h5
TRIGGER_F=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/$SCENE/st_triggers.txt
DST_EIMG_DIR=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/$SCENE/trig_eimgs
WORK_DIR=$(dirname $TRIGGER_F)

if [ ! -e "$EV_F" ]; then
    echo formatting event file for e2calib to
    echo $EV_F
    python data_utils/process_event_format.py -i $EVENT_H5 -o $EV_F -t $TRIGGER_F
fi


if [ ! -e "$DST_EIMG_DIR" ]; then
    echo making eimgs
    echo $DST_EIMG_DIR
    python extrinsics_visualization/make_trig_eimgs.py -i $EV_F -t $TRIGGER_F -o $DST_EIMG_DIR -w $WORK_DIR

fi

python extrinsics_correction/manual_scale_finding.py --scene $SCENE