from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent

"""
play the events live
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
	mv_iterator = EventsIterator(input_path=args.input_path, delta_t=1000)
	height, width = mv_iterator.get_size()  # Camera Geometry

	# Window - Graphical User Interface
	with Window(title="Metavision SDK Get Started", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
		def keyboard_cb(key, scancode, action, mods):
			if action != UIAction.RELEASE:
				return
			if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
				window.set_close_flag()

		window.set_keyboard_callback(keyboard_cb)
		

		accumulation_time_us = 2000
		event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us)

		def on_cd_frame_cb(ts, cd_frame):
			window.show(cd_frame)

		event_frame_gen.set_output_callback(on_cd_frame_cb)

		for evs in mv_iterator:
			# Dispatch system events to the window
			EventLoop.poll_and_dispatch()

			event_frame_gen.process_events(evs)



if __name__ == "__main__":
    main()