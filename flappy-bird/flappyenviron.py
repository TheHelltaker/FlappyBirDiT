import gymnasium as gym
import numpy as np
import pygame
from typing import Optional
from flappy import FlappyBird

class FlappyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        pygame.init()

        self.game = FlappyBird()

        self.action_space = gym.spaces.Discrete(2) # 0 : dont flap, 1 : flap
        self.observation_space = gym.spaces.Dict({
            'bird_pos' : gym.spaces.Box(low = 5, high = 550, shape=(), dtype=np.int32), 
            'next_window' : gym.spaces.Box(
                low=np.array([0, 200], dtype=np.int32),
                high=np.array([467, 683], dtype=np.int32),
            )
        })

        self.prev_score = self.game.score
        self.terminated = True
        self.truncated = False

    def reset(self, seed : Optional[int] = None, options : Optional[dict] = None):
        super.reset(seed=seed)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE))

        state = self.game.get_state_vector()
        self.prev_score = state['score']
        self.terminated = False
        self.truncated = False
        obs = {
            'bird_pos': state['bird_pos'],
            'next_window': np.array(state['next_window'], dtype=np.int32),
        }
        return obs, {}

    def step(self, action):
        if action == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE))

        state = self.game.get_state_vector()
        obs = {
            'bird_pos': state['bird_pos'],
            'next_window': np.array(state['next_window'], dtype=np.int32),
        }
        
        reward = (state['score'] - self.prev_score 
                  - 100.0*(state['terminated'] and not self.terminated)
                  + 0.01*(not state['terminated'])
                  )

        self.prev_score = state['score']
        self.terminated = state['terminated']
        self.truncated = state['truncated']

        return obs, reward, self.terminated, self.truncated, {} 

        


