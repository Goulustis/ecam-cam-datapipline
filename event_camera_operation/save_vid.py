# from curses import window
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent
import cv2
from tqdm import tqdm

"""
Creates a video of n_frames from events
"""

def parse_args():
	import argparse
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description='Metavision SDK Get Started sample.',
										formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		'-i', '--input-raw-file', dest='input_path', default="",
		help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
		"If it's a camera serial number, it will try to open that camera instead.")
	args = parser.parse_args()
	return args

def main():
	
	args = parse_args()

	# Events iterator on Camera or RAW file
	# mv_iterator = EventsIterator(input_path=args.input_path, delta_t=1000)
	mv_iterator = EventsIterator(input_path="./raw_events/rgb_checker/events.h5", delta_t=1000)
	height, width = mv_iterator.get_size()  # Camera Geometry


	accumulation_time_us = 1000
	event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us)

	vid_name = "from_raw.mp4"
	fps = 55 
	time = 30
	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	vid_cap = cv2.VideoWriter(vid_name, fourcc, fps, (width, height))
	

	n_frames = 6000 #fps*time  

	with tqdm(total=n_frames) as pbar:
		def on_cd_frame_cb(ts, cd_frame):
			vid_cap.write(cd_frame)
			pbar.update(1)
			# curr_frame += 1

		event_frame_gen.set_output_callback(on_cd_frame_cb)
		for evs in mv_iterator:
			# Dispatch system events to the window
			if pbar.n >= n_frames:
				break

			EventLoop.poll_and_dispatch()
			event_frame_gen.process_events(evs)

	
	vid_cap.release()




if __name__ == "__main__":
    main()
