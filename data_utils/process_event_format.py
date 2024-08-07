# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# pylint: disable=W0611
"""
Process prosphesee 3.0 event files into 2.0; create input for e2calib for stereo calibaration
"""
"""
h5 io for event storage
you can use 2 compression backends:
    - zlib (fast read, slow write)
    - zstandard (fast read, fast write, but you have to install it)
"""
import sys
sys.path.append(".")

from format_data.format_utils import read_triggers

from tqdm import tqdm
import os.path as osp
import os
import argparse


import h5py
import zlib
try:
    import zstandard
except BaseException:
    pass
import numpy as np

EventCD = np.dtype({'names':['x','y','p','t'], 
                    'formats':['<u2','<u2','i2', '<i8'],
                    'offsets':[0,2,4,8], 
                    'itemsize':16})

class H5EventsWriter(object):
    """
    Compresses & Writes Event Packets as they are read

    Args:
        out_name (str): destination path
        height (int): height of recording
        width (int): width of recording
        compression_backend (str): compression api to be called, defaults to zlib.
        If you can try to use zstandard which is faster at writing.
    """

    def __init__(self, out_name, height, width, compression_backend="zlib"):
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        dt2 = np.int64
        self.f = h5py.File(out_name, "w")
        self.dataset_size_increment = 1000
        shape = (self.dataset_size_increment,)
        self.dset = self.f.create_dataset("event_buffers", shape, maxshape=(None,), dtype=dt)
        self.ts = self.f.create_dataset("event_buffers_start_times", shape, maxshape=(None,), dtype=dt2)
        self.dset.attrs["compression_backend"] = compression_backend
        self.dset.attrs["height"] = height
        self.dset.attrs["width"] = width

        compress_api = globals()[compression_backend]
        self.compress_api = compress_api
        self.index = 0
        self.is_close = False

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def write(self, events):
        """
        Writes event buffer into a compressed packet

        Args:
            events (ndarray): events of type EventCD
        """
        if not len(events):
            return
        if self.index >= len(self.dset):
            new_len = self.dset.shape[0] + self.dataset_size_increment
            self.dset.resize((new_len,))
            self.ts.resize((new_len,))
        self.ts[self.index] = events['t'][0]

        zipped_data = self.compress_api.compress(events)
        zipped_data = np.frombuffer(zipped_data, dtype="uint8")

        self.dset[self.index] = zipped_data
        self.index += 1

    def close(self):
        if not self.is_close:
            self.dset.resize((self.index,))
            self.ts.resize((self.index,))
            self.f.close()
            self.is_close = True

    def __del__(self):
        self.close()


class H5EventsReader(object):
    """
    Reads & Seeks into a h5 file of compressed event packets.

    Args:
        src_name (str): input path
    """

    def __init__(self, path):
        self.path = path
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        dt2 = np.int64
        self.f = h5py.File(path, "r")
        self.len = len(self.f["event_buffers"])
        compression_backend = self.f["event_buffers"].attrs["compression_backend"]
        self.height = self.f["event_buffers"].attrs["height"]
        self.width = self.f["event_buffers"].attrs["width"]
        self.start_times = self.f['event_buffers_start_times'][...]
        self.compress_api = globals()[compression_backend]
        self.start_index = 0
        self.sub_start_index = 0

    def __len__(self):
        return len(self.start_times)

    def seek_in_buffers(self, ts):
        idx = np.searchsorted(self.start_times, ts, side='left')
        return idx

    def seek_time(self, ts):
        idx = np.searchsorted(self.start_times, ts, side='left')
        self.start_index = max(0, idx - 1)
        zipped_data = self.f["event_buffers"][self.start_index]
        unzipped_data = self.compress_api.decompress(zipped_data.data)
        events = np.frombuffer(unzipped_data, dtype=EventCD)
        if self.start_index > 0:
            assert events['t'][0] <= ts
        self.sub_start_index = np.searchsorted(events["t"], ts)
        self.sub_start_index = max(0, self.sub_start_index)

    def get_size(self):
        return (self.height, self.width)

    def __iter__(self):
        for i in range(self.start_index, len(self.f["event_buffers"])):
            zipped_data = self.f["event_buffers"][i]
            unzipped_data = self.compress_api.decompress(zipped_data.data)
            events = np.frombuffer(unzipped_data, dtype=EventCD)
            if i == self.start_index and self.sub_start_index > 0:
                events = events[self.sub_start_index:]
            yield events


def test_read():
    reader = H5EventsReader("test.h5")
    runner = iter(reader)
    return next(runner)


def process_events_h5(inp_file, out_file, st_t=-1, save_np = True):
    reader = H5EventsReader(inp_file)
    runner = iter(reader)

    x, y, t, p = [], [], [], []
    np_events = []
    for event in tqdm(runner, total=len(reader)):
        x.append(event['x'])
        y.append(event['y'])
        t.append(event['t'])
        p.append(event['p'])

        # np_events.append(event)

    concat = lambda x : np.concatenate(x)
    x, y, t, p = concat(x), concat(y), concat(t), concat(p)
    # np_events = concat(np_events)

    cond = t >= st_t
    # x,y,t,p, np_events = [e[cond] for e in [x,y,t,p, np_events]]
    x,y,t,p = [e[cond] for e in [x,y,t,p]]
    # np_events = concat(np_events)
    
    with h5py.File(out_file, "w") as hf:
        hf.create_dataset('x', data=x, shape=x.shape)
        hf.create_dataset('y', data=y, shape=y.shape)
        hf.create_dataset('t', data=t, shape=t.shape)
        hf.create_dataset('p', data=p, shape=p.shape, dtype=np.uint8)
    
    # if save_np:
    #     np_event_path = osp.join(osp.dirname(out_file), "events.npy")
    #     print("saving", np_event_path)
    #     np.save(np_event_path, np_events)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Turn a prophesee 3.0 h5 file to 2.0 h5 file for e2calib')
    parser.add_argument("-i", "--input", help="path to event h5 file", default="data/rgb_checker/events.h5")
    parser.add_argument("-o", "--output", help="path to processed h5 file", default=None)
    # parser.add_argument("-t", "--triggers", help="path to trigger file", default="data/rgb_checker/triggers.txt")
    parser.add_argument("-t", "--triggers", help="path to trigger file", default=None)
    args = parser.parse_args()

    if args.triggers is None:
        st_t = 0
    else:
        triggers = read_triggers(args.triggers)
        st_t = triggers[0]
    
    if args.output is None:
        dir_name = osp.dirname(args.input)
        basename = osp.basename(args.input)
        out_file = osp.join(dir_name, "processed_" + basename)
    else:
        out_file = args.output

    print("processing events", args.input)
    process_events_h5(args.input, out_file)
    print("saving processed events to", out_file)