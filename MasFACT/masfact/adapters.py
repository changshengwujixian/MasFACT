from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .types import AttributedGraph, normalize_measure


class TopologyOptimizer(Protocol):
    def propose(self, support_set: object, attributes: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class TopologyAdapter:
    threshold: float = 0.5

    def from_logits(self, logits: torch.Tensor, attributes: torch.Tensor, measure: torch.Tensor | None = None, weighted: bool = True) -> AttributedGraph:
        adjacency = torch.sigmoid(logits) if weighted else (torch.sigmoid(logits) >= self.threshold).to(logits.dtype)
        adjacency = adjacency * (1.0 - torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype))
        return AttributedGraph(adjacency, attributes, measure)

    def from_probabilities(self, probabilities: torch.Tensor, attributes: torch.Tensor, measure: torch.Tensor | None = None, weighted: bool = True) -> AttributedGraph:
        adjacency = probabilities.float() if weighted else (probabilities >= self.threshold).float()
        adjacency = adjacency * (1.0 - torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype))
        return AttributedGraph(adjacency, attributes, measure)

    def from_discrete(self, adjacency: torch.Tensor, attributes: torch.Tensor, measure: torch.Tensor | None = None) -> AttributedGraph:
        adjacency = adjacency.float() * (1.0 - torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype))
        return AttributedGraph(adjacency, attributes, measure)

    def uniform_measure(self, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return normalize_measure(None, size, device, dtype)
