from __future__ import annotations

import math

import torch

from .posterior import posterior_loss


def pac_bayes_transfer_bound(empirical_risk: torch.Tensor, kl: torch.Tensor, bank_size: int, support_size: int, alignment_cost: torch.Tensor, dispersion: float | torch.Tensor, c_alignment: float, c_dispersion: float, delta: float = 0.05) -> torch.Tensor:
    support_size = max(int(support_size), 1)
    bank_size = max(int(bank_size), 1)
    log_term = math.log((2.0 * bank_size * math.sqrt(support_size)) / max(float(delta), 1e-12))
    complexity = torch.sqrt((kl + empirical_risk.new_tensor(log_term)) / (2.0 * support_size))
    disp = dispersion if torch.is_tensor(dispersion) else empirical_risk.new_tensor(float(dispersion))
    return empirical_risk + complexity + float(c_alignment) * alignment_cost + float(c_dispersion) * disp


def train_loss(empirical_risk: torch.Tensor, posterior_logits: torch.Tensor, prior_logits: torch.Tensor, residual: torch.Tensor, valid_mask: torch.Tensor, lambda_kl: float = 1.0, lambda_residual: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return posterior_loss(empirical_risk, posterior_logits, prior_logits, residual, valid_mask, lambda_kl, lambda_residual)
