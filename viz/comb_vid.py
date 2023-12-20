import glob
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm


def resize_and_pad(frame, target_height, target_width, pad_color=0, axis=0):
    h, w = frame.shape[:2]
    if axis == 0:  # Vertical padding and concatenation
        scale = target_width / w
        new_h, new_w = int(h * scale), target_width
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if new_h < target_height:
            pad_height = (target_height - new_h) // 2
            padded = cv2.copyMakeBorder(resized, pad_height, target_height - new_h - pad_height, 0, 0, cv2.BORDER_CONSTANT, value=pad_color)
        else:
            padded = resized
    else:  # Horizontal padding and concatenation
        scale = target_height / h
        new_h, new_w = target_height, int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if new_w < target_width:
            pad_width = (target_width - new_w) // 2
            padded = cv2.copyMakeBorder(resized, 0, 0, pad_width, target_width - new_w - pad_width, cv2.BORDER_CONSTANT, value=pad_color)
        else:
            padded = resized

    return padded




def read_vid(path):
    if type(path) == str:
        if ".mp4" in path:
            cap = cv2.VideoCapture(path)
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            flag, frames = True, []
            with tqdm(total=n_frames, desc="loading frames") as pbar:
                while flag:
                    flag, frame = cap.read()
                    frames.append(frame)
                    pbar.update(1)
        else:
            frames = [cv2.imread(f) for f in sorted(glob.glob(osp.join(path, "*.png")))]
    elif type(path) == list:
        frames = path
    else:
        assert 0, f"{path} not supported for reading"

    return frames

def concatenate_videos(video_path1, video_path2, output_path=None, axis=1):
    """
    axis = 1 ---> horizontal horizontal concat
    axis = 0 ---> vertical concat
    """
    cap1_ori, cap2_ori = read_vid(video_path1), read_vid(video_path2)

    height1, width1 = cap1_ori[0].shape[:2]
    height2, width2 = cap2_ori[0].shape[:2]
    cap1, cap2 = iter(cap1_ori), iter(cap2_ori)

    target_height = max(height1, height2)
    target_width = max(width1, width2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, 16, (target_width * 2, target_height))
    out=None

    # Determine the total number of frames in the longer video
    total_frames = int(min(len(cap1_ori), len(cap2_ori)))

    frame_idx = 0
    ret1, ret2 = True, True
    comb_frames = []
    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        while True and (frame_idx < total_frames):
            # ret1, frame1 = cap1.read()
            # ret2, frame2 = cap2.read()
            frame1, frame2 = next(cap1), next(cap2)

            # if not ret1 and not ret2:
            #     break
            if (frame1 is None) or (frame2 is None):
                break

            if ret1:
                frame1 = resize_and_pad(frame1, target_height, target_width, axis=axis)
            else:
                frame1 = np.zeros((target_height, target_width, 3), dtype=np.uint8)

            if ret2:
                frame2 = resize_and_pad(frame2, target_height, target_width, axis=axis)
            else:
                frame2 = np.zeros((target_height, target_width, 3), dtype=np.uint8)

            if axis == 1:
                combined_frame = np.hstack((frame1, frame2))
            elif axis == 0:
                combined_frame = np.vstack((frame1, frame2))
            else:
                assert 0, f"{axis} not supported!!"

            if output_path is None:
                comb_frames.append(combined_frame)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined_frame, f"{frame_idx}", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if out is None and (output_path is not None):
                out = cv2.VideoWriter(output_path, fourcc, 16, combined_frame.shape[:2][::-1])
            
            if (output_path is not None):
                out.write(combined_frame)

            pbar.update(1)
            frame_idx += 1

    if len(cap1_ori) != len(cap2_ori):
        print("\033[31m" + f"WARNING: the video size are not the same!! vid1:{len(cap1_ori)}, vid2: {len(cap2_ori)}" + "\033[0m")

    # cap1.release()
    # cap2.release()
    if (output_path is not None):
        out.release()
    
    return comb_frames


# Example usage
if __name__ == "__main__":
    concatenate_videos('/scratch/matthew/projects/ecam-cam-datapipline/tmp/black_seoul_b3_v3_trig_ecamset.mp4', 
                    '/scratch/matthew/projects/ecam-cam-datapipline/tmp/black_seoul_b3_v3_colcam_set.mp4', 
                    '/scratch/matthew/projects/ecam-cam-datapipline/tmp/dev.mp4')
    # concatenate_videos('/scratch/matthew/projects/ecam-cam-datapipline/tmp/comb_vid.mp4', 
    #                 '/scratch/matthew/projects/ecam-cam-datapipline/tmp/comb_vid_ecam.mp4', 
    #                 '/scratch/matthew/projects/ecam-cam-datapipline/tmp/compare_stuff.mp4')
