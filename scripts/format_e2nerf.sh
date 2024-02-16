
### choose scene here; make the workdir first
SCENE=boardroom_b2_v1
N_BINS=4

WORKING_DIR=/ubc/cs/research/kmyi/matthew/backup_copy/raw_real_ednerf_data/work_dir/${SCENE}
TARG_DIR=/ubc/cs/research/kmyi/matthew/projects/E2NeRF/data/real-world/${SCENE}

python format_data/format_e2nerf.py --work_dir ${WORKING_DIR} \
                                    --targ_dir ${TARG_DIR} \
                                    --n_bins ${N_BINS}