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
    [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(cfg.env_id,)) for _ in range(1)]
)

agent = setup_sac(cfg, envs)
agent = load_agent(agent, f"{args.path}/models/{args.checkpoint}")

eval_agent(cfg, agent, envs, stochastic=False, n_eval_episodes=args.n_episodes)
