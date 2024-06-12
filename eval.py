import gymnasium as gym
import numpy as np
import argparse

from main import eval_agent, load_agent
from sac import setup_sac
from utils import init_cfg


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--n_episodes", type=int, default=100)
args = parser.parse_args()

cfg = init_cfg(f"{args.path}/config.yml")

envs = gym.vector.SyncVectorEnv(
    [lambda: gym.wrappers.RecordEpisodeStatistics(
        gym.wrappers.RecordVideo(
            gym.make(cfg.env_id, render_mode="rgb_array"), 
            video_folder=args.path,
            name_prefix="eval",
            episode_trigger=lambda x: x == 0,
        )
    ) for _ in range(1)]
)

agent = setup_sac(cfg, envs)
agent, obs = load_agent(agent, f"{args.path}/models/{args.checkpoint}")

rews = eval_agent(cfg, agent, envs, stochastic=False, n_eval_episodes=args.n_episodes)
rews = np.array(rews)
print(f"Mean reward: {rews.mean()}", f"Std reward: {rews.std()}", f"Min reward: {rews.min()}", f"Max reward: {rews.max()}")
