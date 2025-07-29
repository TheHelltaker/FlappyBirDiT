from game_runner import run_env
from agent import FlappyAgent
import multiprocessing as mp
import gymnasium as gym
import torch

def train_agent(num_iterations, obs_queue, action_queue, stop_event):
    pass


if __name__ == "__main__":
    mp.set_start_method('spawn')
    obs_queue = mp.Queue(10)
    action_queue = mp.Queue(10)
    stop_event = mp.Event()
    environment_process = mp.Process(target=run_env, args=(obs_queue, action_queue))
    agent_process = mp.Process(target=brud, args=(obs_queue, action_queue, stop_event))
    environment_process.start()
    agent_process.start()
    environment_process.join()
    stop_event.set()
    agent_process.join()

