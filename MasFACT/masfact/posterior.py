from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PosteriorConfig:
    temperature: float = 1.0
    threshold: float = 0.5
    eps: float = 1e-8


def masked_logits(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~mask, -30.0)


def edge_probabilities(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(masked_logits(logits, valid_mask)) * valid_mask.to(device=logits.device, dtype=logits.dtype)


def sample_straight_through(logits: torch.Tensor, valid_mask: torch.Tensor, config: PosteriorConfig | None = None) -> torch.Tensor:
    config = config or PosteriorConfig()
    logits = masked_logits(logits, valid_mask)
    uniform = torch.rand_like(logits).clamp(config.eps, 1.0 - config.eps)
    noise = torch.log(uniform) - torch.log1p(-uniform)
    soft = torch.sigmoid((logits + noise) / max(float(config.temperature), config.eps))
    hard = (soft >= float(config.threshold)).to(soft.dtype)
    sample = hard.detach() - soft.detach() + soft
    return sample * valid_mask.to(device=logits.device, dtype=logits.dtype)


def high_probability_topology(logits: torch.Tensor, valid_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = edge_probabilities(logits, valid_mask)
    return (probs >= float(threshold)).to(logits.dtype) * valid_mask.to(device=logits.device, dtype=logits.dtype)


def bernoulli_kl_from_logits(posterior_logits: torch.Tensor, prior_logits: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = valid_mask.to(device=posterior_logits.device, dtype=posterior_logits.dtype)
    q = torch.sigmoid(posterior_logits).clamp(eps, 1.0 - eps)
    p = torch.sigmoid(prior_logits.to(device=posterior_logits.device, dtype=posterior_logits.dtype)).clamp(eps, 1.0 - eps)
    kl = q * (torch.log(q) - torch.log(p)) + (1.0 - q) * (torch.log1p(-q) - torch.log1p(-p))
    denom = torch.clamp(mask.sum(), min=1.0)
    return torch.sum(kl * mask) / denom


def residual_l1(residual: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.to(device=residual.device, dtype=residual.dtype)
    denom = torch.clamp(mask.sum(), min=1.0)
    return torch.sum(torch.abs(residual) * mask) / denom


def posterior_loss(empirical_risk: torch.Tensor, posterior_logits: torch.Tensor, prior_logits: torch.Tensor, residual: torch.Tensor, valid_mask: torch.Tensor, lambda_kl: float, lambda_residual: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kl = bernoulli_kl_from_logits(posterior_logits, prior_logits, valid_mask)
    sparse = residual_l1(residual, valid_mask)
    loss = empirical_risk + float(lambda_kl) * kl + float(lambda_residual) * sparse
    return loss, kl, sparse
