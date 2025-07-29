import threading
import time
import pygame
import multiprocessing as mp
from flappyenviron import FlappyEnv

def polling_thread(env, obs_queue, action_queue):
    """Polls game state and handles action injection"""
    clock = pygame.time.Clock()
    last_action = 0
    while True:
        obs, reward, terminated, truncated, _ = env.step(last_action)
        
        # Send observation to agent
        obs_queue.put({
            'obs': obs,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated
        })
        
        # Check for action from agent
        try:
            action = action_queue.get_nowait()
            if action in [0,1]:
                last_action = action
        except:
            pass  # No action available
            
        # polling rate (60 Hz)
        clock.tick(60)

def run_env(obs_queue, action_queue):
    """Main game runner with threading"""
    # Initialize environment (which initializes game)
    env = FlappyEnv()
    
    # Start polling thread
    poll_thread = threading.Thread(
        target=polling_thread, 
        args=(env, obs_queue, action_queue),
        daemon=True
    )
    poll_thread.start()
    
    # Run main game loop (120 FPS)
    env.game.run()

if __name__ == "__main__":
    obs_queue = mp.Queue()
    action_queue = mp.Queue()
    
    game_process = mp.Process(target=run_env, args=(obs_queue, action_queue))
    game_process.start()
    game_process.join()