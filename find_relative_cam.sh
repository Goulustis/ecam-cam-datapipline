# NOTE: 1) Please select 2 points in 3d before running this script.
    #      The points are used to figure out what the scale between the
    #      relative camera coordinates and the colmap reconstruction
    #   2) install e2calib before running this

# This script will create find the relative positions between two cameras
# run this in project directory (under ./ecam_data_pipeline)


echo sourcing
source ~/.bashrc        # get conda working here
echo sourcing done

# path to standard scene of checker board 9x6
########################## modify inputs here ##############################
WORKING_DIR=$(pwd)
IMAGE_PATH=data/checker/checker_recon/images
EVENT_H5=data/checker/events.h5 
TRIGGER_PATH=$(dirname $EVENT_H5)/triggers.txt 
SQUARE_SIZE=4.23  # size of checkerboard square in desired unit (mm, cm, m)
########################## modify inputs here ##############################
IMAGE_PATH=$(realpath $IMAGE_PATH)
EVENT_H5=$(realpath $EVENT_H5)
TRIGGER_PATH=$(realpath $TRIGGER_PATH)
TARGET_EVENT_H5=$(dirname $EVENT_H5)/processed_$(basename $EVENT_H5)


# CREATE EVENT IMAGES
echo formatting event file for e2calib
python data_utils/process_event_format.py -i $EVENT_H5 -o $TARGET_EVENT_H5

conda activate e2calib           # assuming e2calib is installed
cd third_party/e2calib/python
echo generating event images ...
python offline_reconstruction.py --h5file $TARGET_EVENT_H5 \
                                 --timestamps_file $TRIGGER_PATH
conda deactivate

cd frames
mv e2calib events_imgs
mv events_imgs $(dirname $EVENT_H5)
cd $WORKING_DIR


# RUN STEREO CALIBRATION
# save result to $(dirname $EVENT_H5)/rel_cam.json
REL_CALIB_PATH=$(dirname $EVENT_H5)/rel_calib.json
EVENT_IMG_PATH=$(dirname $EVENT_H5)/events_imgs
echo $EVENT_IMG_PATH
python stereo_calib/camera_calibrate.py -f $IMAGE_PATH -t $EVENT_IMG_PATH -s $SQUARE_SIZE