import numpy as np
from metavision_core.event_io import EventsIterator, H5EventsWriter
from metavision_ml.utils.h5_writer import HDF5Writer
import os.path as osp
import os
from tqdm import tqdm
import time



trigger_type = np.dtype({'names':['p','t','id'], 
                         'formats':['<i2','<i8','<i2'], 
                         'offsets':[0,8,16], 
                         'itemsize':24})

save_dir = "raw_events/calib_checker"
os.makedirs(save_dir, exist_ok=True)
def main():
    sec = 30
    max_time = sec*1e6
    trigger_save_path = osp.join(save_dir, "triggers.npy")
    event_path = osp.join(save_dir, "events.h5")
    trigger_path = osp.join(trigger_save_path)
    e_iter = EventsIterator("", delta_t=10000, max_duration=max_time, **{"use_external_triggers":[0]})

    height, width = e_iter.get_size()
    e_writer = H5EventsWriter(event_path, height=height, width=width)
    # trig_writer = HDF5Writer(trigger_path, "triggers", (), dtype=trigger_type)
    is_rgb_started = False 

    is_printed = False
    trigger_ls = []
    event_ls = []
    pbar = tqdm(total=max_time)
    print("camera running")
    print("saving triggers to", trigger_save_path)
    st_time = time.time()
    for evs in e_iter:
        if evs.size != 0:
            if not is_printed:
                is_printed=True 
                print("event camera running")
            triggers = e_iter.reader.get_ext_trigger_events()
            
            if len(triggers) > 0:
              is_rgb_started = True
              print("seeing triggers")
              # do something with the triggers here
            #   trig_writer.write(triggers)
              # trigger_ls.append(np.array(triggers))
              # e_iter.reader.clear_ext_trigger_events()
        
        if evs.size != 0:
            e_writer.write(evs)
            # event_ls.append(evs)

            pbar.n = evs["t"].max()
            pbar.refresh()

        print("time left", sec - (time.time() - st_time))
    
    e_writer.close()
    # trig_writer.close()
    # trigger_ls = np.concatenate(trigger_ls)
    # np.save(trigger_path, trigger_ls)
    np.save(trigger_path, e_iter.reader.get_ext_trigger_events())
    print("done writing")


if __name__ == "__main__":
    main()
