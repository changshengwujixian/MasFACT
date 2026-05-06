from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from .types import AttributedGraph, FGWResult, PriorAtom


@dataclass
class FGWConfig:
    rho: float = 0.5
    epsilon: float = 0.05
    outer_iterations: int = 20
    sinkhorn_iterations: int = 80
    attribute_metric: Literal["cosine", "sqeuclidean"] = "cosine"
    edge_metric: Literal["sqeuclidean", "absolute"] = "sqeuclidean"
    eps: float = 1e-8


def attribute_cost(current_attributes: torch.Tensor, prototype_attributes: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    current_attributes = current_attributes.float()
    prototype_attributes = prototype_attributes.to(device=current_attributes.device, dtype=current_attributes.dtype)
    if metric == "cosine":
        a = F.normalize(current_attributes, p=2, dim=-1)
        b = F.normalize(prototype_attributes, p=2, dim=-1)
        return (1.0 - a @ b.t()).clamp_min(0.0)
    if metric == "sqeuclidean":
        diff = current_attributes[:, None, :] - prototype_attributes[None, :, :]
        return torch.sum(diff * diff, dim=-1)
    raise ValueError(f"unsupported attribute metric: {metric}")


def relational_cost(scaffold: torch.Tensor, consensus: torch.Tensor, metric: str = "sqeuclidean") -> torch.Tensor:
    scaffold = scaffold.float()
    consensus = consensus.to(device=scaffold.device, dtype=scaffold.dtype)
    diff = scaffold[:, :, None, None] - consensus[None, None, :, :]
    if metric == "sqeuclidean":
        return diff * diff
    if metric == "absolute":
        return torch.abs(diff)
    raise ValueError(f"unsupported edge metric: {metric}")


def entropy_regularizer(coupling: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    value = torch.clamp(coupling, min=eps)
    return torch.sum(value * (torch.log(value) - 1.0))


def fgw_objective(cost_x: torch.Tensor, cost_b: torch.Tensor, coupling: torch.Tensor, config: FGWConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    semantic = torch.sum(cost_x * coupling)
    structural = torch.einsum("ikjl,ij,kl->", cost_b, coupling, coupling)
    entropy = entropy_regularizer(coupling, config.eps)
    total = (1.0 - float(config.rho)) * semantic + float(config.rho) * structural + float(config.epsilon) * entropy
    return total, semantic, structural, entropy


def _log_sinkhorn(cost: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, epsilon: float, iterations: int, eps: float) -> torch.Tensor:
    epsilon = max(float(epsilon), eps)
    log_mu = torch.log(torch.clamp(mu, min=eps))
    log_nu = torch.log(torch.clamp(nu, min=eps))
    kernel = -cost / epsilon
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(int(iterations)):
        u = log_mu - torch.logsumexp(kernel + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(kernel + u[:, None], dim=0)
    return torch.exp(kernel + u[:, None] + v[None, :])


def _linearized_relational_cost(cost_b: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.einsum("ikjl,kl->ij", cost_b, coupling)


def normalize_coupling_rows(coupling: torch.Tensor, mu: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return coupling / torch.clamp(mu, min=eps)[:, None]


def reverse_normalized_coupling(coupling: torch.Tensor, nu: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return coupling.t() / torch.clamp(nu, min=eps)[:, None]


def project_consensus(consensus: torch.Tensor, normalized_coupling: torch.Tensor) -> torch.Tensor:
    return normalized_coupling @ consensus @ normalized_coupling.t()


def solve_fgw(current: AttributedGraph, atom: PriorAtom, config: FGWConfig | None = None) -> FGWResult:
    config = config or FGWConfig()
    atom_graph = atom.graph().to(current.device)
    mu = current.measure.to(device=current.device, dtype=current.dtype)
    nu = atom_graph.measure.to(device=current.device, dtype=current.dtype)
    cost_x = attribute_cost(current.attributes, atom_graph.attributes, config.attribute_metric).to(dtype=current.dtype)
    cost_b = relational_cost(current.adjacency, atom_graph.adjacency, config.edge_metric).to(dtype=current.dtype)
    coupling = torch.outer(mu, nu)
    for _ in range(int(config.outer_iterations)):
        rel = _linearized_relational_cost(cost_b, coupling)
        linear_cost = (1.0 - float(config.rho)) * cost_x + float(config.rho) * rel
        coupling = _log_sinkhorn(linear_cost, mu, nu, config.epsilon, config.sinkhorn_iterations, config.eps)
    total, semantic, structural, entropy = fgw_objective(cost_x, cost_b, coupling, config)
    normalized = normalize_coupling_rows(coupling, mu, config.eps)
    center = project_consensus(atom_graph.adjacency, normalized)
    return FGWResult(
        coupling=coupling,
        alignment_cost=total,
        semantic_cost=semantic,
        structural_cost=structural,
        entropy=entropy,
        normalized_coupling=normalized,
        aligned_center=center,
    )
