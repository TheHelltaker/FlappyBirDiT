import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
from pathlib import Path
import wandb
from wandb.integration.sb3 import WandbCallback


from environ import FlappyBirdEnv
from record import RecorderEnvWrapper


model_dir = os.makedirs(Path(__file__).parent / "agent_models", exist_ok=True)

def make_env(run_id, rank, seed=0, output_dir=None):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param output_dir: (str) path to save recordings
    """
    def _init():
        # Important: use render_mode='rgb_array' for the RecorderEnvWrapper to capture frames.
        env = FlappyBirdEnv(render_mode='rgb_array')
        env.reset(seed=seed + rank)
        
        # Wrap the environment with the recorder
        if output_dir:
            print(f"Recording rollouts for worker {rank} to {output_dir}")
            env = RecorderEnvWrapper(env, output_dir=output_dir, worker_index=rank, run_id=run_id)
        
        return env
    return _init

if __name__ == '__main__':
    # Initialize wandb
    run = wandb.init(
        entity="divyanshtut",
        project="FlappyBirDiT",
        sync_tensorboard=True,  # auto-sync sb3 logs
        monitor_gym=True,       # auto-upload videos
        save_code=True,         # save the main script
    )

    output_path = Path(__file__).parent / "data" / "ppo_rollouts"
    os.makedirs(output_path, exist_ok=True)
    
    n_envs = 2  # Number of parallel environments
    
    # Create the vectorized environment.
    # Using SubprocVecEnv for multiprocessing is recommended for Pygame-based environments.
    print("Creating vectorized environments...")
    env = SubprocVecEnv([make_env(run.id, i, output_dir=output_path) for i in range(n_envs)])

    # Instantiate the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}" # Log to a unique directory for each run
    )

    # Create WandbCallback
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    # Train the model
    print("Starting PPO training...")
    # The total_timesteps is kept low for demonstration. Increase it for proper training.
    model.learn(
        total_timesteps=250000,
        callback=wandb_callback
    )
    print("Training finished.")

    # Save the trained model
    model.save(model_dir / ("agent_" + run.id))
    
    print(f"\nTraining rollouts have been saved in: {output_path}")
    print("To convert a rollout to video, you can use the h5_to_video.py script. For example:")
    print(f"python flappy-bird/h5_to_video.py {output_path}/rollout_worker_0.h5")

    # Clean up the environment
    env.close()
    run.finish()
