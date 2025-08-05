import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
from pathlib import Path
import wandb
from wandb.integration.sb3 import WandbCallback
import signal
import sys
import traceback
import time
import torch

from environ import FlappyBirdEnv
from record import RecorderEnvWrapper

model_dir = Path(__file__).parent / "agent_models"
os.makedirs(model_dir, exist_ok=True)

# Global variables for cleanup
model = None
env = None
run = None

def save_model_emergency(model, run_id, reason="crash"):
    """Emergency model save function"""
    try:
        if model is not None:
            timestamp = int(time.time())
            emergency_path = model_dir / f"emergency_save_{reason}_{run_id}_{timestamp}.zip"
            model.save(emergency_path)
            print(f"\n🚨 Emergency model saved to: {emergency_path}")
            return emergency_path
    except Exception as e:
        print(f"❌ Failed to save emergency model: {e}")
    return None

def cleanup_and_exit(reason="unknown"):
    """Clean up resources and exit gracefully"""
    global model, env, run
    
    print(f"\n🧹 Cleaning up due to: {reason}")
    
    # Save model if available
    if model and run:
        save_model_emergency(model, run.id, reason)
    
    # Close environment
    if env:
        try:
            env.close()
            print("✅ Environment closed")
        except Exception as e:
            print(f"⚠️ Error closing environment: {e}")
    
    # Finish wandb run
    if run:
        try:
            run.finish()
            print("✅ WandB run finished")
        except Exception as e:
            print(f"⚠️ Error finishing WandB run: {e}")
    
    print("🏁 Cleanup completed")

def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C, etc.)"""
    signal_name = signal.Signals(signum).name
    print(f"\n🛑 Received {signal_name} signal")
    cleanup_and_exit(f"signal_{signal_name}")
    sys.exit(0)

def setup_gpu_environment():
    """Configure GPU settings for optimal discrete GPU usage"""
    
    # Force CUDA to use discrete GPU (usually GPU 0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only GPU 0 (discrete GPU)
    
    # Set PyTorch to use CUDA if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)  # Set default device to GPU 0
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return 'cuda:0'
    else:
        print("⚠️ CUDA not available, falling back to CPU")
        return 'cpu'

def make_env(run_id, rank, seed=0, output_dir=None):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param output_dir: (str) path to save recordings
    """
    def _init():
        # Set environment variables for this process to use discrete GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Important: use render_mode='rgb_array' for the RecorderEnvWrapper to capture frames.
        env = FlappyBirdEnv(render_mode='rgb_array')
        env.reset(seed=seed + rank)
        
        # Wrap the environment with the recorder
        if output_dir:
            print(f"Recording rollouts for worker {rank} to {output_dir}")
            env = RecorderEnvWrapper(env, output_dir=output_dir, worker_index=rank, run_id=run_id)
        
        return env
    return _init

class PeriodicSaveCallback(BaseCallback):
    """Custom callback to save model periodically during training"""
    def __init__(self, save_freq=10000, save_path=None, run_id=None, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.run_id = run_id
        self.last_save = 0
        
    def _on_step(self) -> bool:
        """Called by stable-baselines3 after each step"""
        try:
            # Get current timestep
            timestep = self.num_timesteps
            
            # Save every save_freq timesteps
            if timestep - self.last_save >= self.save_freq:
                checkpoint_path = self.save_path / f"checkpoint_{self.run_id}_{timestep}.zip"
                self.model.save(checkpoint_path)
                if self.verbose >= 1:
                    print(f"💾 Checkpoint saved at timestep {timestep}: {checkpoint_path}")
                self.last_save = timestep
        except Exception as e:
            print(f"⚠️ Error in periodic save callback: {e}")
        
        return True  # Continue training

if __name__ == '__main__':
    # Set up GPU environment first
    device = setup_gpu_environment()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    try:
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
        
        n_envs = 1  # Increased from 2 to better utilize GPU
        
        # Create the vectorized environment.
        # Using SubprocVecEnv for multiprocessing is recommended for Pygame-based environments.
        print("Creating vectorized environments...")
        env = SubprocVecEnv([make_env(run.id, i, output_dir=output_path) for i in range(n_envs)])

        # Instantiate the PPO model with explicit device
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,  # Explicitly set device
            tensorboard_log=f"runs/{run.id}" # Log to a unique directory for each run
        )

        # Create WandbCallback
        wandb_callback = WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
        
        # Create periodic save callback
        periodic_save_callback = PeriodicSaveCallback(
            save_freq=10000,  # Save every 10k timesteps
            save_path=model_dir,
            run_id=run.id,
            verbose=1
        )

        # Train the model with error handling
        print("Starting PPO training...")
        print(f"🔄 Automatic checkpoints will be saved every 10,000 timesteps to: {model_dir}")
        
        # Progressive training milestones for Flappy Bird
        # Phase 1: Basic control learning (0-500K)
        # Phase 2: Pipe navigation (500K-1M) 
        # Phase 3: Performance optimization (1M-1.5M)
        TOTAL_TIMESTEPS = 1_000  # 1M steps - good balance of training time vs performance
        
        print(f"🎯 Training for {TOTAL_TIMESTEPS:,} timesteps")
        print("📈 Expected milestones:")
        print("   • 0-100K: Learning basic controls")
        print("   • 100K-500K: Avoiding crashes") 
        print("   • 500K-1M: Mastering pipe navigation")
        
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[wandb_callback, periodic_save_callback]  # Multiple callbacks
        )
        
        print("✅ Training finished successfully!")

        # Save the final trained model
        final_model_path = model_dir / f"agent_final_{run.id}.zip"
        model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")
        
        print(f"\n📁 Training rollouts have been saved in: {output_path}")
        print("To convert a rollout to video, you can use the h5_to_video.py script. For example:")
        print(f"python flappy-bird/h5_to_video.py {output_path}/rollout_worker_0.h5")

        # Clean up the environment
        cleanup_and_exit("normal_completion")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        cleanup_and_exit("keyboard_interrupt")
        
    except OSError as e:
        if e.errno == 28:  # No space left on device
            print(f"\n💽 Disk space error: {e}")
            print("🧹 Consider cleaning up old rollout files:")
            print(f"   rm {output_path}/*.h5")
            cleanup_and_exit("disk_full")
        else:
            print(f"\n💥 OS Error: {e}")
            cleanup_and_exit("os_error")
            
    except Exception as e:
        print(f"\n💥 Unexpected error occurred: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        cleanup_and_exit("unexpected_error")
        
    finally:
        # This ensures cleanup happens no matter what
        if 'model' in locals() or 'env' in locals() or 'run' in locals():
            print("\n🔒 Final cleanup in finally block...")
            # Don't call cleanup_and_exit here to avoid double cleanup
            # Just ensure critical resources are freed
            if 'env' in locals() and env:
                try:
                    env.close()
                except:
                    pass
            if 'run' in locals() and run:
                try:
                    run.finish()
                except:
                    pass