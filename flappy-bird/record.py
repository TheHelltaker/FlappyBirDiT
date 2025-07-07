import mss
import numpy
import h5py
from pynput import keyboard
from pathlib import Path
from multiprocessing import Process, Event
import time

class GameRecorder():
    """
    A class for recording game on the screen and its inputs and converting it into h5 dataset.
    """
    def __init__(self, name : str, window : tuple[int,int,int,int], input_map : list[str]):
        self.name = name
        self.window = window # (top, left, width, height)
        self.input_map = input_map
        self.pressed_keys = set()
        pass

    def record(self, save_path : Path, stop_event=None, fps = 24):
        frame_rate = 1 / fps
        frames = []
        inputs = []
        timestamps = []
        sct = mss.mss()


        start_time = time.time()
        next_frame_time = start_time
        frame_count = 0
        chunk_index = 0


        file_path = save_path / (self.name + ".h5")
        
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        recording = True
        while recording:
            current_time = time.time()
            if current_time >= next_frame_time:
                last_time = time.time()
                timestamps.append(last_time)
                frames.append(numpy.array(sct.grab(self.window)))
                input = {
                    key:key in self.pressed_keys for key in self.input_map
                }
                inputs.append(input)

                frame_count += 1
                next_frame_time = start_time + (frame_count * frame_rate)

                if len(frames) > 1000:
                    Process(target=save_frames, args=(chunk_index, file_path, frames, inputs, timestamps)).start()
                    frames = []
                    inputs = []  
                    timestamps = []
                    chunk_index += 1
            
            if stop_event and stop_event.is_set():
                recording = False
                break

            time.sleep(0.001)
        listener.stop()
        if frames:
            save_frames(chunk_index, file_path, frames, inputs, timestamps)

    def on_press(self, key):
        self.pressed_keys.add(self._key_to_string(key))

    def on_release(self, key):
        self.pressed_keys.discard(self._key_to_string(key))

    def _key_to_string(self, key):
        try:
            return key.char
        except AttributeError:
            return str(key).replace("Key.", "")

def save_frames(chunk_index, file_name, frames, inputs : list[dict[str,bool]], timestamps):
    
    frames = numpy.array(frames, dtype=numpy.uint8)
    timestamps = numpy.array(timestamps, dtype=numpy.float64)

    input_keys = list(inputs[0].keys())
    input_dtype = [(key,'bool') for key in input_keys]

    input_array = numpy.zeros(len(inputs), dtype=input_dtype)
    for i, input in enumerate(inputs):
        for key in input_keys:
            input_array[i][key] = input[key]

    with h5py.File(file_name, 'a') as f:
        chunk_name = f'chunk_{chunk_index:04d}'
        group = f.create_group(chunk_name)

        group.create_dataset('frames',data=frames)
        group.create_dataset('timestamps',data=timestamps)
        group.create_dataset('inputs',data=input_array)

