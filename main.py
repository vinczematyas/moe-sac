import time
import torch
import wandb
import numpy as np
import gymnasium as gym
from safetensors.torch import save_model, load_model
from tqdm import trange
import logging
import os

from sac import train_sac, setup_sac
from utils import init_cfg, fix_seed


def save_agent(path, agent):
    """Save the agent and observations from the replay buffer

    Parameters
    ----------
    path: str
        Path to save the models
    agent: torch.nn.Module
        Agent to save
    """
    os.makedirs(path, exist_ok=True)
    save_model(agent.actor, f"{path}/actor.safetensors")
    save_model(agent.qf1, f"{path}/qf1.safetensors")
    save_model(agent.qf2, f"{path}/qf2.safetensors")
    save_model(agent.qf1_target, f"{path}/qf1_target.safetensors")
    save_model(agent.qf2_target, f"{path}/qf2_target.safetensors")
    np.savez_compressed(f"{path}/observations.npz", array=agent.rb.observations)


def load_agent(agent, path):
    """Load the agent and observations from the replay buffer

    Parameters
    ----------
    agent: torch.nn.Module
        Agent to load
    path: str
        Path to the models

    Returns
    -------
    agent: torch.nn.Module
        Loaded agent
    obs: np.ndarray
        Loaded observations
    """
    assert os.path.exists(path), f"Path {path} does not exist"
    load_model(agent.actor, f"{path}/actor.safetensors")
    load_model(agent.qf1, f"{path}/qf1.safetensors")
    load_model(agent.qf2, f"{path}/qf2.safetensors")
    load_model(agent.qf1_target, f"{path}/qf1_target.safetensors")
    load_model(agent.qf2_target, f"{path}/qf2_target.safetensors")
    obs = np.load(f"{path}/observations.npz")["array"]  
    return agent, obs


def eval_agent(cfg, agent, envs, stochastic=True, tree=None, n_eval_episodes=10, feature_subset=None):
    """Evaluate the agent

    Parameters
    ----------
    cfg: class 'utils.NestedDictToNamespace'
        Configuration
    agent: torch.nn.Module
        Agent to evaluate
    envs: gym.vector.SyncVectorEnv
        Environment used for evaluation
    tree: DecisionTreeClassifier, optional
        Tree used to infer the expert
    n_eval_episodes: int, optional
        Number of evaluation episodes

    Returns
    -------
    episode_rewards: list
        Rewards obtained in each evaluation episode
    """

    obs, _ = envs.reset()
    episode_rewards = []

    with torch.no_grad():
        while len(episode_rewards) < n_eval_episodes:
            obs = torch.Tensor(obs).to(cfg.device)
            if tree:
                if feature_subset:
                    obs_tree = obs[:, feature_subset]
                else:
                    obs_tree = obs
                # infer tree to get expert idx
                leaf_idxs = int(tree.predict(obs_tree)[0])
                mean_expert, log_std_expert = agent.actor.mean_experts[leaf_idxs], agent.actor.log_std_experts[leaf_idxs]
                # infer the expert to get mean and log_std
                mean, log_std = mean_expert(obs), log_std_expert(obs)
                log_std = torch.tanh(log_std)
                log_std = agent.actor.LOG_STD_MIN + 0.5 * (agent.actor.LOG_STD_MAX - agent.actor.LOG_STD_MIN) * (log_std + 1)
                # get action using mean and log_std
                actions = agent.actor.get_action(obs, mean, log_std, training=False)
            else:
                actions = agent.actor.get_action(obs, training=False)
            if stochastic:
                actions = actions[0].cpu().detach().numpy()
            else:
                actions = actions[-1].cpu().detach().numpy()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    episode_rewards.append(info["episode"]["r"][0])
                    print(f"episode {len(episode_rewards)}/{n_eval_episodes} reward: {info['episode']['r'][0]}")

            obs = next_obs

    return episode_rewards


def train_sac_agent(cfg, agent, envs):
    """Train the agent

    Parameters
    ----------
    cfg: class 'utils.NestedDictToNamespace'
        Configuration
    agent: torch.nn.Module
        Agent to train
    envs: gym.vector.SyncVectorEnv
        Environment used for training
    """

    # fill RB with random data
    obs, _ = envs.reset(seed=cfg.seed)
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

    obs, _ = envs.reset(seed=cfg.seed+1)
    generator = trange(cfg.total_timesteps, desc="Training")
    for global_step in generator:
        actions = agent.actor.get_action(torch.Tensor(obs).to(cfg.device))
        actions = actions[0].cpu().detach().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                agent.actor.gate.capacity = torch.zeros(cfg.sac.n_experts)
                if cfg.log.wandb:
                    wandb.log({
                    "global_step": global_step,
                    "episodic_return": info['episode']['r'][0],
                    "episodic_length": info['episode']['l'][0],
                })
                if cfg.log.log_local:
                    logging.info(f"{global_step}, {info['episode']['r'][0]}, {info['episode']['l'][0]}")
                generator.set_postfix({"return": round(info['episode']['r'][0], 3), "length": info['episode']['l'][0]})

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        agent.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        train_sac(cfg, agent)

        if cfg.log.save_models and (global_step+1) % int(1e5) == 0:
            save_agent(f"{cfg.run_path}/models/checkpoint_{str(global_step+1)[:-3]}k", agent)


def create_routing_dataset(agent, obs, n_samples, tree=None):
    """Create a dataset for observation - routing network mapping

    Parameters
    ----------
    agent: torch.nn.Module
        Agent used to collect data
    obs: np.ndarray
        Observations
    n_samples: int
        Number of samples to collect
    tree: DecisionTreeClassifier, optional
        Tree used to infer the expert

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
        curr_obs = obs[obs_data.shape[0]:obs_data.shape[0]+curr_n_samples]
        curr_obs = torch.Tensor(curr_obs)
        if tree:
            expert_idxs = torch.Tensor(tree.predict(curr_obs))
        else:
            expert_idxs = agent.actor.gate(curr_obs, training=False)[1]
        obs_data = torch.cat((obs_data, curr_obs), dim=0)
        idx_data = torch.cat((idx_data, expert_idxs), dim=0)
    return obs_data, idx_data


def main(cfg):
    """Main function to train the agent

    Parameters
    ----------
    cfg: class 'utils.NestedDictToNamespace'
        Configuration
    """

    n_envs = 1  # for now, we only support single env training
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(cfg.env_id,)) for _ in range(n_envs)]
    )

    fix_seed(cfg.seed, envs)

    agent = setup_sac(cfg, envs)
    train_sac_agent(cfg, agent, envs)

    if cfg.log.save_models:
        save_agent(f"{cfg.run_path}/models/checkpoint_final", agent)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # run args
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--config", type=str, default="walker.yml")
    parser.add_argument("--device", type=str)
    parser.add_argument("--total_timesteps", type=int)
    parser.add_argument("--seed", type=int)
    # logging args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log_local", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    # sac args
    parser.add_argument("--n_experts", type=int)
    parser.add_argument("--topk", type=int)
    args = parser.parse_args()

    cfg = init_cfg(f"configs/{args.config}")

    cfg.update(**vars(args))
    cfg.log.update(**vars(args))
    cfg.sac.update(**vars(args))

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

