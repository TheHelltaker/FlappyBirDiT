from __future__ import annotations
import multiprocessing as mp
import time
import numpy as np
import torch
import tqdm

class FlappyAgent:
    def __init__(self, learning_rate : float, 
                 init_eps : float, eps_decay : float, final_eps :float, device='cpu'):


        self.net = Net(device=device)

        self.lr = learning_rate
        self.epsilon = init_eps
        self.decay_rate = eps_decay
        self.final_eps = final_eps

        self.training_error = []

    def get_action(self, obs):
        if np.random.random_sample() < self.epsilon:
            return np.random.choice([0, 1])
        else:
            logits = self.net(torch.cat(
                [torch.tensor([obs['bird_pos']]),torch.from_numpy(obs['next_window'])],
                dim = 0
            ))
            return torch.argmax(logits).item()

    def decay_epsilon(self):
        self.epsilon = max(self.final_eps, self.epsilon - self.decay_rate)


    def run_agent(self, num_episodes : int, obs_queue : mp.Queue, action_queue : mp.Queue, train : bool = True, stop_event=None,
                  epsilon : float | None = None):

        if not train:
            self.net.load_state_dict(torch.load('agent_weights.pth', weights_only=True))
            self.net.eval()
        else:
            self.net.train()
    
        agent_epsilon = self.epsilon
        if epsilon is not None:
            self.epsilon = epsilon

        for episode in tqdm(range(num_episodes)):
            if stop_event and stop_event.is_set():
                break
            while True:
                try:
                    state = obs_queue.get_nowait()
                    if state['terminated'] or state['truncated']:
                        continue
                    obs = state['obs']
                    action_queue.put(
                        self.get_action(obs)
                    )
                except:
                    pass
                time.sleep(1.0/ 60.0) #(60Hz)

        self.epsilon = agent_epsilon
            

        

class Net(torch.nn.Module):
    def __init__(self, device : torch.device):
        super().__init__()
        self.ff1 = torch.nn.Linear(3, 5, device=device)
        self.ac1 = torch.nn.ReLU()
        self.ff2 = torch.nn.Linear(5, 8, device=device)
        self.ac2 = torch.nn.ReLU()
        self.ff3 = torch.nn.Linear(8, 5, device=device)
        self.ac3 = torch.nn.ReLU()
        self.ff4 = torch.nn.Linear(5, 2, device=device)

    def forward(self, x):
        x = self.ac1(self.ff1(x))
        x = self.ac2(self.ff2(x))
        x = self.ac3(self.ff3(x))
        x = self.ff4(x)

        return x



    