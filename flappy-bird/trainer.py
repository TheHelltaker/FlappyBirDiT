import ray
from ray import tune, train
from ray.rllib.algorithms.dqn import DQNConfig

from ray.tune.registry import register_env

from ray.air.integrations.wandb import WandbLoggerCallback

from pprint import pprint
from pathlib import Path

from environ import FlappyBirdEnv
from record import RecorderEnvWrapper

curpath = Path(__file__).parent

def env_creator(env_config):
    env = FlappyBirdEnv(render_mode="human")
    return RecorderEnvWrapper(
        env,
        output_dir=env_config["output_dir"],
        worker_index=env_config.worker_index
    )
register_env("flappy_bird_env", env_creator)


config = (
    DQNConfig()
    .environment(
        env="flappy_bird_env",
        env_config={
            "output_dir": str(curpath / "data") # Use string concatenation for path
        },
    )
    .training(
        gamma=0.99,
        lr=1e-4,
        train_batch_size=32,
        target_network_update_freq=1000,
        epsilon=[
            [0, 1.0],
            [25, 0.75],
            [50, 0.5],
            [90, 0.0]
        ]
    )
    .env_runners(
        num_env_runners=2,
        enable_connectors=False,
    )
)

ray.init()

dqn = config.build_algo()

for _ in range(100):
    results =dqn.train()
    pprint(results['training_iteration'])

ray.shutdown()