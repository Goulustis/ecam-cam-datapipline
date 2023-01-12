PNT_PATH="data/checker/checker_recon/sparse/0/points3D.bin"
C_CAM_PATH="data/checker/checker_recon/sparse/0/images_mtx.npy"
E_CAM_PATH="data/checker/e_cams.npy"

python viz/viz_relative_mtx.py -c $C_CAM_PATH -e $E_CAM_PATH -p $PNT_PATH