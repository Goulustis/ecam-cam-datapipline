from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
import numpy as np
from tqdm import tqdm

import os
import os.path as osp

# Run camera and listen for triggers 

def main():
    max_time = 30*1e6
    e_iter = EventsIterator("", delta_t=10000, max_duration=max_time, **{"use_external_triggers":[0]})
    print("camera running")
    
    trig_ls = []
    pbar = tqdm(total=max_time)
    for evs in e_iter:
        if evs.size != 0:
            print("non-zero events")
            pbar.n = evs["t"][-1]
            pbar.refresh()
            triggers = e_iter.reader.get_ext_trigger_events()

            if len(triggers) > 0:
              
              # print(triggers)
              print("trigger received")
              
              # do something with triggers
              # e_iter.reader.clear_ext_trigger_events()
    dev_dir = "raw_events/dev_trig"
    os.makedirs(dev_dir, exist_ok=True)
    np.save(osp.join(dev_dir, "triggers.npy"), e_iter.reader.get_ext_trigger_events())

if __name__ == "__main__":
    main()
