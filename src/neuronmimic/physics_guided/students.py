"""Three comparable Student dynamics baselines."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .teacher import CableTeacherConfig


class _BaseStudent(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = int(n)

    def step(self, state: Tensor, event: Tensor) -> Tensor:
        raise NotImplementedError

    def rollout(self, initial_state: Tensor, events: Tensor) -> Tensor:
        states = [initial_state]
        state = initial_state
        for t in range(events.shape[0]):
            event = events[t] if events[t].ndim == 2 else events[t].unsqueeze(0)
            state = self.step(state, event)
            states.append(state)
        return torch.stack(states)


class MLPDynamics(_BaseStudent):
    """Pure MLP one-step map; morphology is deliberately not explicit."""
    def __init__(self, n: int, hidden: int = 96) -> None:
        super().__init__(n)
        self.net = nn.Sequential(nn.Linear(3 * n, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2 * n))

    def step(self, state: Tensor, event: Tensor) -> Tensor:
        return state + self.net(torch.cat([state, event], dim=-1))


class GraphDynamics(_BaseStudent):
    """Message-passing Student that retains the morphology adjacency."""
    def __init__(self, adjacency: Tensor, hidden: int = 64) -> None:
        n = int(adjacency.shape[0])
        super().__init__(n)
        norm = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adjacency", norm)
        # The graph model predicts bounded voltage and conductance increments;
        # the adjacency is the only explicit morphology input.
        self.node = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def step(self, state: Tensor, event: Tensor) -> Tensor:
        v = state[:, :self.n]
        g = state[:, self.n:]
        neighbor = torch.einsum("ij,bj->bi", self.adjacency, v)
        features = torch.stack([v, g, event, neighbor], dim=-1)
        delta = 0.1 * torch.tanh(self.node(features))
        return torch.cat([v + delta[..., 0] + 0.1 * (neighbor - v), g + delta[..., 1]], dim=-1)


class PhysicsResidualDynamics(_BaseStudent):
    """Known low-order cable update plus a bounded learned residual."""
    def __init__(self, adjacency: Tensor, config: CableTeacherConfig, hidden: int = 48) -> None:
        super().__init__(int(adjacency.shape[0]))
        self.config = config
        self.register_buffer("adjacency", adjacency.to(torch.float32))
        self.residual = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def step(self, state: Tensor, event: Tensor) -> Tensor:
        n = self.n
        v, g = state[:, :n], state[:, n:]
        dt = self.config.dt_ms
        g_next = g * torch.exp(torch.tensor(-dt / self.config.tau_syn_ms, device=state.device)) + event * self.config.synapse_increment
        axial = self.config.axial_conductance * (torch.einsum("ij,bj->bi", self.adjacency, v) - self.adjacency.sum(dim=-1) * v)
        effective_leak = self.config.leak_conductance * self.config.capacitance / self.config.tau_mem_ms
        leak = effective_leak * (self.config.resting_voltage - v)
        syn = g_next * (self.config.synapse_reversal - v)
        physics = v + dt / self.config.capacitance * (leak + axial + syn)
        residual = 0.1 * torch.tanh(self.residual(torch.stack([v, g, event], dim=-1))).squeeze(-1)
        return torch.cat([physics + residual, g_next], dim=-1)
