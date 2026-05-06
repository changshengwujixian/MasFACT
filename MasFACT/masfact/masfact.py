from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .fgw import FGWConfig
from .memory import ConsolidationConfig, PriorBank, RetrievalConfig
from .objectives import pac_bayes_transfer_bound, train_loss
from .posterior import PosteriorConfig, edge_probabilities, high_probability_topology, sample_straight_through
from .types import AdaptationOutput, AttributedGraph, PosteriorState, PriorAtom


@dataclass
class MasFACTConfig:
    fgw: FGWConfig = field(default_factory=FGWConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    lambda_kl: float = 1.0
    lambda_residual: float = 0.1
    c_alignment: float = 1.0
    c_dispersion: float = 1.0
    delta: float = 0.05


class MasFACT:
    def __init__(self, prior_bank: PriorBank | None = None, config: MasFACTConfig | None = None):
        self.prior_bank = prior_bank or PriorBank()
        self.config = config or MasFACTConfig()

    def initialize_atom(self, graph: AttributedGraph, utility: float = 0.0, dispersion: float = 0.0) -> None:
        self.prior_bank.add(PriorAtom(graph.adjacency.detach(), graph.attributes.detach(), graph.measure.detach(), utility, dispersion, 1))

    def adapt(self, current: AttributedGraph, residual: torch.Tensor, valid_mask: torch.Tensor | None = None) -> AdaptationOutput:
        retrieval = self.prior_bank.retrieve(current, self.config.fgw, self.config.retrieval)
        center = retrieval.fgw.aligned_center.to(current.device, current.dtype)
        residual = residual.to(current.device, current.dtype)
        if valid_mask is None:
            valid_mask = torch.ones_like(center)
            valid_mask = valid_mask * (1.0 - torch.eye(center.shape[0], device=center.device, dtype=center.dtype))
        valid_mask = valid_mask.to(current.device, current.dtype)
        posterior_logits = center + residual
        prior_prob = edge_probabilities(center, valid_mask)
        posterior_prob = edge_probabilities(posterior_logits, valid_mask)
        execution = high_probability_topology(posterior_logits, valid_mask, self.config.posterior.threshold)
        posterior = PosteriorState(center, posterior_logits, residual, valid_mask)
        return AdaptationOutput(retrieval, posterior, prior_prob, posterior_prob, execution)

    def differentiable_sample(self, output: AdaptationOutput) -> torch.Tensor:
        return sample_straight_through(output.posterior.posterior_logits, output.posterior.valid_mask, self.config.posterior)

    def loss(self, empirical_risk: torch.Tensor, output: AdaptationOutput) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return train_loss(empirical_risk, output.posterior.posterior_logits, output.posterior.prior_logits, output.posterior.residual, output.posterior.valid_mask, self.config.lambda_kl, self.config.lambda_residual)

    def bound(self, empirical_risk: torch.Tensor, kl: torch.Tensor, output: AdaptationOutput, support_size: int) -> torch.Tensor:
        return pac_bayes_transfer_bound(empirical_risk, kl, len(self.prior_bank), support_size, output.retrieval.fgw.alignment_cost, output.retrieval.atom.dispersion, self.config.c_alignment, self.config.c_dispersion, self.config.delta)

    def consolidate(self, current: AttributedGraph, output: AdaptationOutput, utility: float, transfer_complexity: float | None = None) -> str:
        if transfer_complexity is None:
            kl = torch.mean(torch.abs(output.posterior_prob - output.prior_prob)).item()
            transfer_complexity = kl + self.config.c_alignment * float(output.retrieval.fgw.alignment_cost.detach().cpu()) + self.config.c_dispersion * float(output.retrieval.atom.dispersion)
        return self.prior_bank.consolidate(current, output.retrieval, output.posterior_prob.detach(), utility, float(transfer_complexity), self.config.consolidation)
