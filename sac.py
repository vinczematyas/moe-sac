import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple
from stable_baselines3.common.buffers import ReplayBuffer
from typing import List
import math


class TopkRouter(nn.Module):
    """Top-k router for Sparse-MoE"""

    def __init__(self, input_dim: int, n_experts: int, topk: int = 1):
        """Initialize the Top-k router

        Parameters
        ----------
        input_dim: int
            The dimension of the input
        n_experts: int
            The number of experts
        topk: int
            The number of experts to select
        """

        super(TopkRouter, self).__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.fc = nn.Linear(input_dim, n_experts)
        self.noise = torch.distributions.Normal(
            loc=torch.tensor([0.0]*n_experts), scale=torch.tensor([1.0/n_experts]*n_experts)
        )
        # load balancing
        self.importance = torch.zeros(n_experts)
        self.load = torch.zeros(n_experts)

    def forward(self, x, training):
        """Forward pass of the Top-k router

        Softmax before top-k
        Noise injection for exploration
        Auxiliary loss calculation for load balancing
        same as: https://proceedings.neurips.cc/paper/2021/file/48237d9f2dea8c74c2a72126cf63d933-Paper.pdf

        Parameters
        ----------
        x: torch.Tensor
            The input tensor
        training: bool
            Whether the model is training or not
        """

        logits = self.fc(x)

        if training:
            noisy_logits = logits + self.noise.sample()

            importance = F.softmax(noisy_logits, dim=-1).sum(0)
            self.importance = (torch.std(importance)/torch.mean(importance))**2

            threshold = torch.max(noisy_logits, dim=-1).values
            load = (1 - self.noise.cdf(threshold.unsqueeze(1) - logits)).sum(0)
            self.load = (torch.std(load)/torch.mean(load))**2
        else:
            noisy_logits = logits

        noisy_logits = F.softmax(noisy_logits, dim=-1)
        topk_logits, topk_indices = torch.topk(noisy_logits, self.topk, dim=-1)
        sparse_logits = torch.zeros_like(logits).scatter_(index=topk_indices, src=topk_logits, dim=-1)
        return sparse_logits, topk_indices


class Actor(nn.Module):
    def __init__(self, cfg, env):
        super().__init__()
        self.cfg = cfg.sac
        self.observation_shape = np.prod(env.single_observation_space.shape)
        self.action_shape = np.prod(env.single_action_space.shape)

        self.mean_experts = nn.ModuleList([nn.Linear(self.observation_shape, self.action_shape) for _ in range(self.cfg.n_experts)])
        self.log_std_experts = nn.ModuleList([nn.Linear(self.observation_shape, self.action_shape) for _ in range(self.cfg.n_experts)])
        self.gate = TopkRouter(
            input_dim=self.observation_shape, 
            n_experts=self.cfg.n_experts, 
            topk=self.cfg.topk, 
        )

        # action scaling and bias
        action_space = env.single_action_space
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        ) 
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

    def forward(self, x, training):
        x = x.float()
        # initialize mean and log_std
        mean = torch.zeros(x.size(0), self.action_shape).to(x.device)
        log_std = torch.zeros(x.size(0), self.action_shape).to(x.device)

        flat_x = x.view(-1, x.size(-1))

        # infer gating network
        gating_output, gating_indices = self.gate(x, training)
        flat_gating_output = gating_output.view(-1, gating_output.shape[-1])

        for i in range(self.cfg.n_experts):
            expert_mask = (gating_indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            # if expert is selected
            if flat_mask.any():
                expert_input = flat_x[flat_mask]

                # infer expert
                mean_i = self.mean_experts[i](expert_input)
                log_std_i = self.log_std_experts[i](expert_input)

                # extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weigted_mean = mean_i * gating_scores
                weigted_log_std = log_std_i * gating_scores

                # update final output
                mean[expert_mask] += weigted_mean.squeeze(1)
                log_std[expert_mask] += weigted_log_std.squeeze(1)

        # apply action scaling and bias
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x, mean=None, log_std=None, training=False):
        if mean == None and log_std == None:
            mean, log_std = self(x, training)
        std = log_std.exp()
        x_t = torch.randn_like(mean) * std + mean
        y_t = torch.tanh(x_t)  # normalize action to [-1, 1]
        action = y_t * self.action_scale + self.action_bias  # scale action to environment's range
        log_prob = -0.5 * ((x_t - mean) / std).pow(2) - std.log() - 0.5 * math.log(2 * math.pi)  # gaussian log likelihood
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)  # adjustment for Tanh squashing
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SoftQNetwork(nn.Module):
    def __init__(self, cfg, env):
        super().__init__()

        input_dim = np.prod(env.single_observation_space.shape) + np.prod(env.single_action_space.shape)
        in_dims = [input_dim] + [256] * cfg.sac.q_depth
        out_dims = [256] * cfg.sac.q_depth + [1]

        self.fc_list = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, out_dims)])

    def forward(self, x, a): 
        x = x.float()
        x = torch.cat([x, a], 1)

        for i, fc_layer in enumerate(self.fc_list[:-1]):
            x = F.relu(fc_layer(x))
        x = self.fc_list[-1](x)

        return x

SACComponents = namedtuple("SACComponents", ["actor", "qf1", "qf2", "qf1_target", "qf2_target", "q_optimizer", "actor_optimizer", "rb", "target_entropy", "log_alpha", "a_optimizer", "counter"])

def setup_sac(cfg, env):
    actor = Actor(cfg, env).to(cfg.device)
    qf1 = SoftQNetwork(cfg, env).to(cfg.device)
    qf2 = SoftQNetwork(cfg, env).to(cfg.device)
    qf1_target = SoftQNetwork(cfg, env).to(cfg.device)
    qf2_target = SoftQNetwork(cfg, env).to(cfg.device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.sac.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=cfg.sac.policy_lr)

    if cfg.sac.alpha_auto == True:
        target_entropy = -torch.prod(torch.tensor(env.single_action_space.shape).to(cfg.device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=cfg.device)
        a_optimizer = optim.Adam([log_alpha], lr=cfg.sac.q_lr)
    else:
        target_entropy = None
        log_alpha = None
        a_optimizer = None

    # MinMax Replay Buffer so we can add new best or worst experiences
    rb = ReplayBuffer(
        cfg.sac.buffer_size,
        env.single_observation_space,
        env.single_action_space,
        cfg.device,
        handle_timeout_termination=False,
    )

    counter = {'n_steps': 0}

    return SACComponents(actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, rb, target_entropy, log_alpha, a_optimizer, counter)

def train_sac(cfg, sac):
    if cfg.sac.alpha_auto == True:
        alpha = sac.log_alpha.exp().item()
    else:
        alpha = cfg.sac.alpha

    data = sac.rb.sample(cfg.sac.batch_size)
    with torch.no_grad():
        next_state_actions, next_state_log_pi, _ = sac.actor.get_action(data.next_observations)
        qf1_next_target = sac.qf1_target(data.next_observations, next_state_actions)
        qf2_next_target = sac.qf2_target(data.next_observations, next_state_actions)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * cfg.sac.gamma * (min_qf_next_target).view(-1)

    qf1_a_values = sac.qf1(data.observations, data.actions).view(-1)
    qf2_a_values = sac.qf2(data.observations, data.actions).view(-1)
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    # optimize the model
    sac.q_optimizer.zero_grad()
    qf_loss.backward()
    sac.q_optimizer.step()

    if sac.counter['n_steps'] % cfg.sac.policy_frequency == 0:  # TD 3 Delayed update support
        for _ in range(cfg.sac.policy_frequency):  # compensate for the delay in policy updates
            pi, log_pi, _ = sac.actor.get_action(data.observations, training=True)
            qf1_pi = sac.qf1(data.observations, pi)
            qf2_pi = sac.qf2(data.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

            aux_loss = 0.5 * sac.actor.gate.importance + 0.5 * sac.actor.gate.load
            actor_loss += 0.01 * aux_loss

            sac.actor_optimizer.zero_grad()
            actor_loss.backward()
            sac.actor_optimizer.step()

            if cfg.sac.alpha_auto == True:
                with torch.no_grad():
                    _, log_pi, _ = sac.actor.get_action(data.observations)
                alpha_loss = (-sac.log_alpha.exp() * (log_pi + sac.target_entropy)).mean()

                sac.a_optimizer.zero_grad()
                alpha_loss.backward()
                sac.a_optimizer.step()
                alpha = sac.log_alpha.exp().item()


    if sac.counter['n_steps'] % cfg.sac.target_network_frequency == 0:
        for param, target_param in zip(sac.qf1.parameters(), sac.qf1_target.parameters()):
            target_param.data.copy_(cfg.sac.tau * param.data + (1 - cfg.sac.tau) * target_param.data)
        for param, target_param in zip(sac.qf2.parameters(), sac.qf2_target.parameters()):
            target_param.data.copy_(cfg.sac.tau * param.data + (1 - cfg.sac.tau) * target_param.data)

    # pruning
    prune_end = cfg.sac.prune_end * cfg.total_timesteps
    prune_start = cfg.sac.prune_start * cfg.total_timesteps
    if sac.counter['n_steps'] > prune_start and sac.counter['n_steps'] < prune_end:
        prune_amount = cfg.sac.prune_percent * (1 - (1 - (sac.counter['n_steps'] - prune_start) / (prune_end - prune_start)) ** 3)
    elif sac.counter['n_steps'] >= prune_end:
        prune_amount = cfg.sac.prune_percent
    else:
        prune_amount = 0.0

    prune.ln_structured(sac.actor.gate.fc, 'weight', amount=prune_amount, n=1, dim=-1)
    prune.remove(agent.actor.gate.fc, 'weight')

    sac.counter['n_steps']  += 1

