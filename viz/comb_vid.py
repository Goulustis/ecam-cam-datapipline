import cv2
import numpy as np
from tqdm import tqdm

def resize_and_pad(frame, target_height, target_width, pad_color=0):
    h, w = frame.shape[:2]
    scale = target_height / h
    new_h, new_w = target_height, int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    if new_w < target_width:
        pad_width = (target_width - new_w) // 2
        padded = cv2.copyMakeBorder(resized, 0, 0, pad_width, target_width - new_w - pad_width, cv2.BORDER_CONSTANT, value=pad_color)
    else:
        padded = resized

    return padded

def concatenate_videos(video_path1, video_path2, output_path):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_height = max(height1, height2)
    target_width = max(width1, width2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, 16, (target_width * 2, target_height))
    out=None

    # Determine the total number of frames in the longer video
    total_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    frame_idx = 0
    # Using tqdm for the progress bar
    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 and not ret2:
                break

            if ret1:
                frame1 = resize_and_pad(frame1, target_height, target_width)
            else:
                frame1 = np.zeros((target_height, target_width, 3), dtype=np.uint8)

            if ret2:
                frame2 = resize_and_pad(frame2, target_height, target_width)
            else:
                frame2 = np.zeros((target_height, target_width, 3), dtype=np.uint8)

            combined_frame = np.hstack((frame1, frame2))
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, 16, combined_frame.shape[:2][::-1])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined_frame, f"{frame_idx}", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            out.write(combined_frame)
            

            pbar.update(1)
            frame_idx += 1

    cap1.release()
    cap2.release()
    out.release()


# Example usage
concatenate_videos('/scratch/matthew/projects/ecam-cam-datapipline/tmp/black_seoul_b1_v3_trig_ecamset.mp4', 
                   '/scratch/matthew/projects/ecam-cam-datapipline/tmp/black_seoul_b1_v3_colcam_set.mp4', 
                   '/scratch/matthew/projects/ecam-cam-datapipline/tmp/comb_vid.mp4')
