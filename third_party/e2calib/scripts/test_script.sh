bash
conda activate e2calib
cd python
python offline_reconstruction.py  --h5file  /scratch/matthew/projects/e2calib/data/grid_test.h5 --freq_hz 5 --upsample_rate 4 --height 720 --width 1280
