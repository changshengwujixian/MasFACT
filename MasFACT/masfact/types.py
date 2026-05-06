from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x if torch.is_floating_point(x) else x.float()


def normalize_measure(measure: Optional[torch.Tensor], size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if measure is None:
        return torch.full((size,), 1.0 / max(size, 1), device=device, dtype=dtype)
    value = _as_float_tensor(measure).to(device=device, dtype=dtype).flatten()
    if value.numel() != size:
        raise ValueError(f"measure has {value.numel()} entries but expected {size}")
    value = torch.clamp(value, min=0.0)
    total = value.sum()
    if float(total.detach().cpu()) <= 1e-12:
        return torch.full((size,), 1.0 / max(size, 1), device=device, dtype=dtype)
    return value / total


@dataclass
class AttributedGraph:
    adjacency: torch.Tensor
    attributes: torch.Tensor
    measure: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.adjacency = _as_float_tensor(self.adjacency)
        self.attributes = _as_float_tensor(self.attributes).to(device=self.adjacency.device, dtype=self.adjacency.dtype)
        if self.adjacency.dim() != 2 or self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if self.attributes.dim() != 2 or self.attributes.shape[0] != self.adjacency.shape[0]:
            raise ValueError("attributes must be a matrix with one row per node")
        self.measure = normalize_measure(self.measure, self.num_nodes, self.adjacency.device, self.adjacency.dtype)

    @property
    def num_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def device(self) -> torch.device:
        return self.adjacency.device

    @property
    def dtype(self) -> torch.dtype:
        return self.adjacency.dtype

    def to(self, device: torch.device | str) -> "AttributedGraph":
        return AttributedGraph(
            adjacency=self.adjacency.to(device),
            attributes=self.attributes.to(device),
            measure=self.measure.to(device) if self.measure is not None else None,
        )


@dataclass
class PriorAtom:
    consensus: torch.Tensor
    attributes: torch.Tensor
    measure: Optional[torch.Tensor] = None
    utility: float = 0.0
    dispersion: float = 0.0
    support_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.consensus = _as_float_tensor(self.consensus)
        self.attributes = _as_float_tensor(self.attributes).to(device=self.consensus.device, dtype=self.consensus.dtype)
        if self.consensus.dim() != 2 or self.consensus.shape[0] != self.consensus.shape[1]:
            raise ValueError("consensus must be a square matrix")
        if self.attributes.dim() != 2 or self.attributes.shape[0] != self.consensus.shape[0]:
            raise ValueError("attributes must be a matrix with one row per prototype node")
        self.measure = normalize_measure(self.measure, self.num_nodes, self.consensus.device, self.consensus.dtype)
        self.utility = float(self.utility)
        self.dispersion = float(self.dispersion)
        self.support_count = int(max(1, self.support_count))

    @property
    def num_nodes(self) -> int:
        return int(self.consensus.shape[0])

    def graph(self) -> AttributedGraph:
        return AttributedGraph(self.consensus, self.attributes, self.measure)

    def detached(self) -> "PriorAtom":
        return PriorAtom(
            consensus=self.consensus.detach().clone(),
            attributes=self.attributes.detach().clone(),
            measure=self.measure.detach().clone() if self.measure is not None else None,
            utility=self.utility,
            dispersion=self.dispersion,
            support_count=self.support_count,
            metadata=dict(self.metadata),
        )


@dataclass
class FGWResult:
    coupling: torch.Tensor
    alignment_cost: torch.Tensor
    semantic_cost: torch.Tensor
    structural_cost: torch.Tensor
    entropy: torch.Tensor
    normalized_coupling: torch.Tensor
    aligned_center: torch.Tensor


@dataclass
class RetrievalResult:
    atom_index: int
    atom: PriorAtom
    fgw: FGWResult
    retrieval_score: torch.Tensor


@dataclass
class PosteriorState:
    prior_logits: torch.Tensor
    posterior_logits: torch.Tensor
    residual: torch.Tensor
    valid_mask: torch.Tensor


@dataclass
class AdaptationOutput:
    retrieval: RetrievalResult
    posterior: PosteriorState
    prior_prob: torch.Tensor
    posterior_prob: torch.Tensor
    execution_topology: torch.Tensor
