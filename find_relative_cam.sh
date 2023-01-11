# NOTE: Please select 2 points in 3d before running this script.
#       The points are used to figure out what the scale between the
#       relative camera coordinates and the colmap reconstruction

# This script will create find the relative positions between two cameras
# run this in project directory (under ./ecam_data_pipeline)


# path to standard scene of checker board 9x6
WORKING_DIR=$(pwd)
IMAGE_PATH=some_path1
EVENT_H5=some_path2/name.h5 
TRIGGER_PATH=some_path2/triggers.txt 
SQUARE_SIZE=4.23  # size of checkerboard square in desired unit (mm, cm, m)

TARGET_EVENT_H5=$(dirname $EVENT_H5)/processed_$(basename $EVENT_H5).h5
REL_CALIB_PATH=$(dirname $EVENT_H5)/rel_calib.json


# CREATE EVENT IMAGES
python data_utils/process_event_format.py -i $EVENT_H5 -o $TARGET_EVENT_H5

conda activate e2calib           # assuming e2calib is installed
cd third_party/e2calib/python
python offline_reconstruction.py --h5file $TARGET_EVENT_H5 \
                                 --timestamps_file $TRIGGER_PATH


cd frames
mv e2calib events_imgs
mv events_imgs $(dirname $EVENT_H5)
cd $WORKING_DIR


# RUN STEREO CALIBRATION
# save result to $(dirname $EVENT_H5)/rel_cam.json
EVENT_IMG_PATH=$(dirname $EVENT_H5)/event_imgs
python stereo_calib/camera_calibrate.py -f $IMAGE_PATH -t $EVENT_IMG_PATH -s $SQUARE_SIZE