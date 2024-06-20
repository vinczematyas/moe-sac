import gymnasium as gym
import numpy as np
import torch
import cv2

from main import load_agent
from sac import setup_sac


def eval_agent(cfg, path, checkpoint, n_episodes=100, return_results=False):
    envs = gym.vector.SyncVectorEnv([lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(cfg.env_id),) for _ in range(1)])
    agent = setup_sac(cfg, envs)
    agent, _, _ = load_agent(agent, f"{path}/models/{checkpoint}")

    env = gym.wrappers.RecordEpisodeStatistics(
            gym.wrappers.RecordVideo(
                gym.make(cfg.env_id, render_mode="rgb_array"),
                video_folder="videos",
                name_prefix=path.split("/")[-1],
                episode_trigger=lambda x: x == 0,
            )
    )

    if return_results:
        mean_obs = [[] for _ in range(cfg.sac.n_experts)]
        mean_actions = [[] for _ in range(cfg.sac.n_experts)]
        n_expert_selections = [0] * cfg.sac.n_experts
        saved_renders = [[] for _ in range(cfg.sac.n_experts)]

    episodic_rewards = []
    episodic_lengths = []

    obs, _ = env.reset()

    episode_index = 0
    n_steps = 0
    while episode_index < n_episodes:
        with torch.no_grad():
            n_steps += 1

            expert_idx = agent.actor.gate(torch.Tensor(obs).to(cfg.device), training=False)[1].item()
            action = agent.actor.mean_experts[expert_idx](torch.Tensor(obs).to(cfg.device)).detach().numpy()

            next_obs, rewards, terminations, truncations, infos = env.step(action)

            if return_results:
                mean_obs[int(expert_idx)].append(obs)
                mean_actions[int(expert_idx)].append(action)
                render_small = cv2.resize(cv2.resize(env.render(), (256, 256))[70:250, 50:200], (100, 120))
                saved_renders[int(expert_idx)].append(render_small)
                n_expert_selections[int(expert_idx)] += 1

            obs = next_obs

            if terminations or truncations:
                episode_index += 1
                print(f"episode {episode_index}/{n_episodes} reward: {infos['episode']['r'][0]}")
                episodic_rewards.append(infos["episode"]["r"][0])
                episodic_lengths.append(infos["episode"]["l"][0])
                obs, _ = env.reset()

    if return_results:
        mean_obs_final = [np.mean(np.stack(i), axis=0) if i else [] for i in mean_obs]
        mean_actions_final = [np.mean(np.stack(i), axis=0) if i else [] for i in mean_actions]

        closest_obs_idx = []
        for expert_idx in range(cfg.sac.n_experts):
            if type(mean_obs_final[expert_idx]) == np.ndarray:
                closest_obs_idx.append(np.argmin([np.linalg.norm(np.array(obs) - np.array(mean_obs_final[expert_idx])) for obs in mean_obs[expert_idx]]))
            else:
                closest_obs_idx.append(None)

        mean_render = [saved_renders[expert_idx][closest_obs_idx[expert_idx]] if closest_obs_idx[expert_idx] else [] for expert_idx in range(cfg.sac.n_experts)]

        n_expert_selections = np.array(n_expert_selections) / n_steps
        mean_obs_final = np.array([i if type(i) == np.ndarray else np.zeros(env.observation_space.shape) for i in mean_obs_final])
        mean_actions_final = np.array([i if type(i) == np.ndarray else np.zeros(env.action_space.shape) for i in mean_actions_final])
        mean_render = np.array([i if type(i) == np.ndarray else np.zeros((120, 100, 3)) for i in mean_render])

    episodic_rewards = np.array(episodic_rewards)
    episodic_lengths = np.array(episodic_lengths)

    print(f"\nEVAL \n\tmean: {np.mean(episodic_rewards):.3f} \n\tstd: {np.std(episodic_rewards):.3f} \n\tmin: {np.min(episodic_rewards):.3f} \n\tmax: {np.max(episodic_rewards):.3f}\n")

    if return_results:
        return mean_obs_final, mean_actions_final, mean_render, n_expert_selections, episodic_rewards, episodic_lengths
    else:
        return episodic_rewards, episodic_lengths


if __name__ == "__main__":
    import argparse
    from utils import init_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--n_episodes", type=int, default=100)
    args = parser.parse_args()

    cfg = init_cfg(f"{args.path}/config.yml")

    eval_agent(cfg, args.path, args.checkpoint, args.n_episodes)

