import mss
import numpy as np
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
                frames.append(np.array(sct.grab(self.window)))
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
    
    frames = np.array(frames, dtype=np.uint8)
    timestamps = np.array(timestamps, dtype=np.float64)

    input_keys = list(inputs[0].keys())
    input_dtype = [(key,'bool') for key in input_keys]

    input_array = np.zeros(len(inputs), dtype=input_dtype)
    for i, input in enumerate(inputs):
        for key in input_keys:
            input_array[i][key] = input[key]

    with h5py.File(file_name, 'a') as f:
        chunk_name = f'chunk_{chunk_index:04d}'
        group = f.create_group(chunk_name)

        group.create_dataset('frames',data=frames)
        group.create_dataset('timestamps',data=timestamps)
        group.create_dataset('inputs',data=input_array)

import gymnasium as gym

class RecorderEnvWrapper(gym.Wrapper):
    """
    A wrapper for a Gymnasium environment that records gameplay to an H5 file.
    """
    def __init__(self, env, output_dir, worker_index, run_id):
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.worker_index = worker_index
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.file_path = self.output_dir / f"rollout_worker_{self.worker_index}_run_{run_id}.h5"
        self.chunk_index = 0
        self.buffer = []
        self.chunk_size = 1000
        self.action_meanings = self.env.get_action_meanings()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        frame = self.env.render()
        timestamp = time.time()
        
        self.buffer.append((frame, action, timestamp))

        if len(self.buffer) >= self.chunk_size:
            self._save_chunk()

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        if self.buffer:
            self._save_chunk()
        self.env.close()

    def _save_chunk(self):
        if not self.buffer:
            return

        frames, actions, timestamps = zip(*self.buffer)
        self.buffer = []

        frames = np.array(frames, dtype=np.uint8)
        timestamps = np.array(timestamps, dtype=np.float64)

        input_keys = [self.action_meanings[1]]
        input_dtype = [(key, 'bool') for key in input_keys]
        input_array = np.zeros(len(actions), dtype=input_dtype)
        for i, action in enumerate(actions):
            if action == 1:
                input_array[i][input_keys[0]] = True

        with h5py.File(self.file_path, 'a') as f:
            chunk_name = f'chunk_{self.chunk_index:04d}'
            group = f.create_group(chunk_name)
            group.create_dataset('frames', data=frames)
            group.create_dataset('timestamps', data=timestamps)
            group.create_dataset('inputs', data=input_array) # Save as 'inputs'
        
        self.chunk_index += 1

