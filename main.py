import time
import torch
import wandb
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from safetensors.torch import save_model, load_model
from tqdm import trange
import logging
import os
import dtreeviz
import pandas as pd

from sac import train_sac, setup_sac
from utils import init_cfg


def save_models(agent, tree, path):
    """Save the agent and tree models

    Attributes
    ----------
    agent: torch.nn.Module
        Agent to save
    tree: sklearn.tree.DecisionTreeClassifier
        Decision tree to save
    path: str
        Path to save the models
    """
    os.makedirs(path, exist_ok=True)
    save_model(agent.actor, f"{path}/actor.safetensors")
    save_model(agent.qf1, f"{path}/qf1.safetensors")
    save_model(agent.qf2, f"{path}/qf2.safetensors")
    save_model(agent.qf1_target, f"{path}/qf1_target.safetensors")
    save_model(agent.qf2_target, f"{path}/qf2_target.safetensors")
    pickle.dump(tree, open(f"{path}/tree.pkl", "wb"))


def load_models(agent, path):
    """Load the agent and tree models

    Attributes
    ----------
    agent: torch.nn.Module
        Agent to load
    path: str
        Path to the models

    Returns
    -------
    agent: torch.nn.Module
        Loaded agent
    tree: sklearn.tree.DecisionTreeClassifier
        Loaded decision tree
    """
    assert os.path.exists(path), f"Path {path} does not exist"
    load_model(agent.actor, f"{path}/actor.safetensors")
    load_model(agent.qf1, f"{path}/qf1.safetensors")
    load_model(agent.qf2, f"{path}/qf2.safetensors")
    load_model(agent.qf1_target, f"{path}/qf1_target.safetensors")
    load_model(agent.qf2_target, f"{path}/qf2_target.safetensors")
    tree = pickle.load(open(f"{path}/tree.pkl", "rb"))
    return agent, tree


def eval_agent(agent, envs, tree=None, n_eval_episodes=10):
    """Evaluate the agent in the environment

    Attributes
    ----------
    agent: torch.nn.Module
        Agent to evaluate
    envs: gym.vector.SyncVectorEnv
        Environment used for evaluation
    n_eval_episodes: int
        Number of episodes to evaluate the agent

    Returns
    -------
    episode_rewards: list
        List of rewards obtained in each episode
    """

    obs, _ = envs.reset()
    episode_rewards = []

    while len(episode_rewards) < n_eval_episodes:
        obs = torch.Tensor(obs).to(cfg.device)
        if tree:
            # infer tree to get expert idx
            leaf_idxs = int(tree.predict(obs)[0])
            mean_expert, log_std_expert = agent.actor.mean_experts[leaf_idxs], agent.actor.log_std_experts[leaf_idxs]
            # infer the expert to get mean and log_std
            mean, log_std = mean_expert(obs), log_std_expert(obs)
            log_std = torch.tanh(log_std)
            log_std = agent.actor.LOG_STD_MIN + 0.5 * (agent.actor.LOG_STD_MAX - agent.actor.LOG_STD_MIN) * (log_std + 1)
            # get action using mean and log_std
            actions = agent.actor.get_action(obs, mean, log_std, training=False)
        else:
            actions = agent.actor.get_action(obs, training=False)
        actions = actions[0].cpu().detach().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                episode_rewards.append(info["episode"]["r"][0])

        obs = next_obs

    return episode_rewards


def train_sac_agent(cfg, agent, envs):
    """Train the agent in the environment

    Attributes
    ----------
    cfg: dataclasses.dataclass
        Configuration dataclass
    agent: torch.nn.Module
        Agent to train
    envs: gym.vector.SyncVectorEnv
        Environment used for training
    n_train_steps: int
        Number of training steps
    """

    # fill RB with random data
    obs, _ = envs.reset()
    generator = trange(int(1e4), desc="Filling RB")
    for _ in generator:
        actions = envs.action_space.sample()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        agent.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

    obs, _ = envs.reset()
    generator = trange(cfg.total_timesteps, desc="Training")
    for global_step in generator:
        actions = agent.actor.get_action(torch.Tensor(obs).to(cfg.device))
        actions = actions[0].cpu().detach().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if cfg.log.wandb:
                    wandb.log({
                    "global_step": global_step,
                    "episodic_return": info['episode']['r'][0],
                    "episodic_length": info['episode']['l'][0],
                })
                if cfg.log.log_local:
                    logging.info(f"{global_step}, {info['episode']['r'][0]}, {info['episode']['l'][0]}")
                generator.set_postfix({f"return": round(info['episode']['r'][0], 3), "length": info['episode']['l'][0]})
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        agent.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        train_sac(cfg, agent)


def create_routing_dataset(agent, envs, n_samples, rb_available=True, tree=None):
    """Create a dataset for observation - routing network mapping

    Attributes
    ----------
    agent: torch.nn.Module
        Agent used to collect data
    n_samples: int
        Number of samples to collect

    Returns
    -------
    obs_data: torch.Tensor
        Observations
    idx_data: torch.Tensor
        Leaf indices
    """

    obs_data = torch.Tensor()
    idx_data = torch.Tensor()
    while obs_data.shape[0] < n_samples:
        curr_n_samples = min(250, n_samples - obs_data.shape[0])
        if rb_available:
            data = agent.rb.sample(curr_n_samples)
            obs = data.observations
        else:
            obs = torch.cat([torch.Tensor(envs.observation_space.sample()) for _ in range(curr_n_samples)], dim=0)
        obs = obs.float()
        if tree:
            leaf_idxs = torch.Tensor(tree.predict(obs))
        else:
            leaf_idxs = agent.actor.gate(obs)[1]
        obs_data = torch.cat((obs_data, obs), dim=0)
        idx_data = torch.cat((idx_data, leaf_idxs), dim=0)
    return obs_data, idx_data


def main(cfg):
    n_envs = 1  # for now, we only support single env training
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(cfg.env_id,)) for _ in range(n_envs)]
    )

    agent = setup_sac(cfg, envs)
    train_sac_agent(cfg, agent, envs)

    print(f"Creating dataset for DT training")
    X, y = create_routing_dataset(agent, envs, n_samples=cfg.dt.n_ds_samples)

    dt = DecisionTreeClassifier(max_depth=cfg.dt.max_depth)

    print(f"Fitting decision tree on {min(cfg.total_timesteps+10000, cfg.dt.n_ds_samples)} samples")
    dt.fit(X, y)

    # evaluate the agent
    print(f"Evaluating on 10 episodes")
    print(eval_agent(agent, envs))
    print(f"Evaluation on 10 episodes w. tree")
    print(eval_agent(agent, envs, tree=dt))
    
    if cfg.log.save_models:
        save_models(agent, dt, f"{cfg.run_path}/models")
        agent, dt = load_models(agent, f"{cfg.run_path}/models")

        print(f"Evaluation on 10 episodes w. tree after re-loading")
        print(eval_agent(agent, envs, tree=dt))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # run args
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--config", type=str, default="hopper.yml")
    parser.add_argument("--device", type=str)
    parser.add_argument("--total_timesteps", type=int)
    # logging args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log_local", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    # sac args
    parser.add_argument("--n_experts", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--q_depth", type=int)
    # decsion tree args
    parser.add_argument("--n_ds_samples", type=int)
    parser.add_argument("--max_depth", type=int)
    args = parser.parse_args()

    cfg = init_cfg(f"configs/{args.config}")
    cfg.update(**vars(args))
    cfg.log.update(**vars(args))
    cfg.sac.update(**vars(args))
    cfg.dt.update(**vars(args))

    run_path = f"{args.config.split('.')[0]}/{args.run_name}_{time.strftime('%m%d_%H%M')}"
    cfg.run_path = "runs/" + run_path 
    os.makedirs(cfg.run_path, exist_ok=True)

    cfg.to_yaml(f"{cfg.run_path}/config.yml")

    if cfg.log.log_local:
        logging.basicConfig(
            filename=f"{cfg.run_path}/log.log", 
            level=logging.INFO, format="%(message)s"
        )
    if cfg.log.wandb:
        wandb.init(project="moe-sac", name=run_path)

    main(cfg)

