# code to format the data into a dataset usuable by algorithm
# if using the structure detailed in README. Only [REQUIRED] ones are needed to be put in
source ~/.bashrc
conda activate ecam_proc

WORKING_DIR=$(pwd)
##################### general config for both dataset ############################
SCENE_PATH=data/rgb_checker                  # [REQUIRED] path/scene
TARG_DIR=data/formatted_rgb_checker          # [REQUIRED] path to location to save the formatted dataset
CREATE_EIMG=False                            # [REQUIRED] True for creating event images, False for just the camera extrinsics
RELCAM_PATH=$SCENE_PATH/rel_cam.json         # path to relative camera
TRIGGER_PATH=$SCENE_PATH/triggers.txt        # path to triggers
ECAM_TARG_DIR=$TARG_DIR/ecam_set             # path to save processed event camera data

echo formatting event camera dataset

python format_data/format_ecam_set.py --scene_path $SCENE_PATH \
                                      --relcam_path $RELCAM_PATH \
                                      --targ_dir $ECAM_TARG_DIR \
                                      --trigger_path $TRIGGER_PATH \
                                      --create_eimgs $CREATE_EIMG



##################### config for color camera dataset #####################
SCENE=$(basename $SCENE_PATH)
IMG_PATH=$SCENE_PATH/${SCENE}_recon/images        # path to original image
COLMAP_PATH=$SCENE_PATH/${SCENE}_recon/sparse/0    # path to sparse colmap output
COLCAM_TARG_DIR=$TARG_DIR/colcam_set               # path to save save the color camera data
TRIG_IDS_PATH=$ECAM_TARG_DIR/trig_ids.npy          # path to trigger ids when formatting event camera set
BLURRY_PER=0                                       # [REQUIRED] parameter to filter blurry images or not; 0 to turn off
IMG_SCALE=1                                        # [REQUIRED] scale of used on colmap (eg if used 2x downsampled of original images, then this is 2)


echo formating color camera dataset

python format_data/format_col_set.py --img_dir $IMG_PATH \
                                     --colmap_dir $COLMAP_PATH \
                                     --img_scale $IMG_SCALE \
                                     --target_dir $COLCAM_TARG_DIR \
                                     --blurry_per $BLURRY_PER \
                                     --trigger_path $TRIGGER_PATH \
                                     --trig_ids_path $TRIG_IDS_PATH

######## copy scene.json from colcam_set to ecam_set ######3
cd $WORKING_DIR
cp $COLCAM_TARG_DIR/scene.json $ECAM_TARG_DIR

######## copy the images over ##############
cp -r $IMG_PATH $COLCAM_TARG_DIR
cd $COLCAM_TARG_DIR
mv $(basename $IMG_PATH) 1x
mkdir rgb
mv 1x rgb

######### MISC #########
cd $WORKING_DIR
cp $RELCAM_PATH $TARG_DIR
