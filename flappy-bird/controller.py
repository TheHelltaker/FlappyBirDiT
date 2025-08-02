from flappy import FlappyBird
from record import GameRecorder
import multiprocessing as mp
from pathlib import Path
import time

def run_game():
    game = FlappyBird()
    game.run()

def run_recorder(recording_name, save_path, stop_event):
    window = FlappyBird.game_window()
    input_map = FlappyBird.game_inputs()
    recorder = GameRecorder(recording_name,window,input_map)

    recorder.record(save_path, stop_event)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    stop_event = mp.Event()
    recording_name = "recorder-testing-1"
    game_process = mp.Process(target=run_game)
    data_dir = Path(__file__).parent / "data/"
    recorder_process = mp.Process(target=run_recorder, args=(recording_name,data_dir,stop_event))
    game_process.start()
    time.sleep(2)
    recorder_process.start()
    game_process.join()
    stop_event.set()
    recorder_process.join()
