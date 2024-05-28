import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(MLP, self).__init__()
        self.fc_list = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims + [output_dim])]
        )

    def forward(self, x):
        for fc in self.fc_list[:-1]:
            x = F.relu(fc(x))
        x = self.fc_list[-1](x)
        return x


class TopkRouter(nn.Module):
    def __init__(self, input_dim: int, n_experts: int, topk: int = 1, router_hidden_dims: List[int] = []):
        super(TopkRouter, self).__init__()
        self.fc = MLP(input_dim, router_hidden_dims, n_experts)
        self.fc_noise = MLP(input_dim, router_hidden_dims, n_experts)
        self.topk = topk

        # TODO: add noise to the logits using another MLP (https://github.com/davidmrau/mixture-of-experts)

    def forward(self, x, training=False):
        """Infers the top-k experts for the input x.

        Take topk experts with the highest logits
        Make the logits of the other experts to be -inf
        Take softmax to get the routing probabilities

        Returns
        -------
        routing_probs: torch.Tensor
            The routing probabilities of the experts
        topk_idx: torch.Tensor
            The indices of the top-k experts
        """
        logits = self.fc(x)
        if training:
            noise_logits = self.fc_noise(x)
            noise = torch.randn_like(logits) + F.softplus(self.fc_noise(x))
            logits += noise  # load balancing w. noise  ISSUE: ruins performance
        topk_logits, topk_indices = torch.topk(logits, self.topk, dim=-1)
        zeros = torch.full_like(logits, fill_value=float('-inf'))
        sparse_logits = zeros.scatter_(index=topk_indices, src=topk_logits, dim=-1)
        return F.softmax(sparse_logits, dim=-1), topk_indices

