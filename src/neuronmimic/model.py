"""A small, differentiable, connectome-preserving LIF calibration layer.

This model is intended for small subgraphs and smoke tests. The validated
large-network LIF backend remains the production event simulator. A soft spike
surrogate is used only for fitting; hard events are returned for inspection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class NeuronMimicConfig:
    dt_ms: float = 0.1
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    v_rest_mv: float = -52.0
    v_threshold_mv: float = -45.0
    v_reset_mv: float = -52.0
    surrogate_beta: float = 8.0
    residual_rank: int = 0
    residual_scale_mv: float = 0.5
    parameter_prior_scale: float = 0.05


class NeuronMimic(nn.Module):
    """Directed LIF backbone with bounded per-neuron calibration parameters.

    ``edge_src`` and ``edge_dst`` are never symmetrized. ``edge_weight`` is an
    effective signed coupling supplied by the caller, not a measured conductance.
    External current is ``[time, neuron]`` or ``[time, batch, neuron]``.
    """

    def __init__(self, n_neurons: int, edge_src: Tensor, edge_dst: Tensor, edge_weight: Tensor, config: NeuronMimicConfig | None = None) -> None:
        super().__init__()
        self.config = config or NeuronMimicConfig()
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        if not (edge_src.ndim == edge_dst.ndim == edge_weight.ndim == 1):
            raise ValueError("edge arrays must be one-dimensional")
        if not (edge_src.numel() == edge_dst.numel() == edge_weight.numel()):
            raise ValueError("directed edge arrays must have equal length")
        if edge_src.numel() and (int(edge_src.min()) < 0 or int(edge_dst.min()) < 0 or int(edge_src.max()) >= n_neurons or int(edge_dst.max()) >= n_neurons):
            raise ValueError("edge index outside neuron range")
        self.n_neurons = int(n_neurons)
        self.register_buffer("edge_src", edge_src.to(torch.long))
        self.register_buffer("edge_dst", edge_dst.to(torch.long))
        self.register_buffer("edge_weight", edge_weight.to(torch.float32))
        self.log_tau_mem_delta = nn.Parameter(torch.zeros(n_neurons))
        self.log_tau_syn_delta = nn.Parameter(torch.zeros(n_neurons))
        self.threshold_delta = nn.Parameter(torch.zeros(n_neurons))
        self.gain_delta = nn.Parameter(torch.zeros(n_neurons))
        rank = int(self.config.residual_rank)
        if rank > 0:
            self.residual_basis = nn.Parameter(torch.zeros(n_neurons, rank))
            self.residual_input = nn.Parameter(torch.zeros(rank, n_neurons))
            nn.init.normal_(self.residual_basis, std=0.01)
            nn.init.normal_(self.residual_input, std=0.01)
        else:
            self.register_parameter("residual_basis", None)
            self.register_parameter("residual_input", None)

    def baseline_state(self, batch_size: int = 1, *, device: Optional[torch.device] = None) -> tuple[Tensor, Tensor]:
        device = device or self.edge_weight.device
        return (
            torch.full((batch_size, self.n_neurons), self.config.v_rest_mv, device=device),
            torch.zeros(batch_size, self.n_neurons, device=device),
        )

    def effective_parameters(self) -> dict[str, Tensor]:
        c = self.config
        return {
            "tau_mem_ms": c.tau_mem_ms * torch.exp(0.25 * self.log_tau_mem_delta),
            "tau_syn_ms": c.tau_syn_ms * torch.exp(0.25 * self.log_tau_syn_delta),
            "threshold_mv": c.v_threshold_mv + 2.0 * torch.tanh(self.threshold_delta),
            "gain": 1.0 + 0.25 * torch.tanh(self.gain_delta),
        }

    def _residual_current(self, external_current: Tensor) -> Tensor:
        if self.residual_basis is None:
            return torch.zeros_like(external_current)
        coeff = torch.tanh(external_current @ self.residual_input.T)
        return self.config.residual_scale_mv * (coeff @ self.residual_basis.T)

    def forward(self, external_current: Tensor, *, initial_voltage: Optional[Tensor] = None, initial_synaptic: Optional[Tensor] = None) -> dict[str, Tensor]:
        if external_current.ndim == 2:
            external_current = external_current[:, None, :]
        if external_current.ndim != 3 or external_current.shape[-1] != self.n_neurons:
            raise ValueError("external_current must have shape [time, neuron] or [time, batch, neuron]")
        t_steps, batch, _ = external_current.shape
        params = self.effective_parameters()
        v, syn = self.baseline_state(batch, device=external_current.device)
        if initial_voltage is not None:
            v = initial_voltage
        if initial_synaptic is not None:
            syn = initial_synaptic
        tau_m = params["tau_mem_ms"].to(external_current.device).clamp_min(1e-3)
        tau_s = params["tau_syn_ms"].to(external_current.device).clamp_min(1e-3)
        gain = params["gain"].to(external_current.device)
        threshold = params["threshold_mv"].to(external_current.device)
        alpha_m = torch.exp(-self.config.dt_ms / tau_m)
        alpha_s = torch.exp(-self.config.dt_ms / tau_s)
        residual = self._residual_current(external_current.reshape(t_steps * batch, self.n_neurons)).reshape(t_steps, batch, self.n_neurons)
        voltages, soft_spikes, hard_spikes = [], [], []
        for t in range(t_steps):
            syn = alpha_s * syn
            v = self.config.v_rest_mv + (v - self.config.v_rest_mv) * alpha_m + (1.0 - alpha_m) * gain * (external_current[t] + residual[t] + syn)
            soft = torch.sigmoid(self.config.surrogate_beta * (v - threshold))
            hard = (v >= threshold).to(v.dtype)
            voltages.append(v)
            soft_spikes.append(soft)
            hard_spikes.append(hard)
            v = v - soft * (v - self.config.v_reset_mv)
            syn_next = torch.zeros_like(syn)
            syn_next.index_add_(1, self.edge_dst, soft[:, self.edge_src] * self.edge_weight[None, :])
            syn = syn_next
        return {"voltage_mv": torch.stack(voltages), "spike_soft": torch.stack(soft_spikes), "spike_hard": torch.stack(hard_spikes), "parameters": params}

    def prior_penalty(self) -> Tensor:
        terms = [self.log_tau_mem_delta, self.log_tau_syn_delta, self.threshold_delta, self.gain_delta]
        if self.residual_basis is not None:
            terms.extend([self.residual_basis, self.residual_input])
        return self.config.parameter_prior_scale * sum(torch.mean(x.square()) for x in terms)
