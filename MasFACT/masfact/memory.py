from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .fgw import FGWConfig, reverse_normalized_coupling, solve_fgw
from .types import AttributedGraph, PriorAtom, RetrievalResult


@dataclass
class RetrievalConfig:
    lambda_dispersion: float = 0.1
    lambda_utility: float = 0.1


@dataclass
class ConsolidationConfig:
    utility_threshold: float = 0.0
    complexity_threshold: float = 1.0
    update_rate: float = 0.25
    max_atoms: Optional[int] = None


class PriorBank:
    def __init__(self, atoms: Optional[List[PriorAtom]] = None):
        self.atoms = list(atoms or [])

    def __len__(self) -> int:
        return len(self.atoms)

    def add(self, atom: PriorAtom) -> None:
        self.atoms.append(atom.detached())

    def retrieve(self, current: AttributedGraph, fgw_config: FGWConfig | None = None, retrieval_config: RetrievalConfig | None = None) -> RetrievalResult:
        if len(self.atoms) == 0:
            raise ValueError("prior bank is empty")
        fgw_config = fgw_config or FGWConfig()
        retrieval_config = retrieval_config or RetrievalConfig()
        best: RetrievalResult | None = None
        for index, atom in enumerate(self.atoms):
            result = solve_fgw(current, atom, fgw_config)
            score = result.alignment_cost + float(retrieval_config.lambda_dispersion) * result.alignment_cost.new_tensor(atom.dispersion) - float(retrieval_config.lambda_utility) * result.alignment_cost.new_tensor(atom.utility)
            if best is None or float(score.detach().cpu()) < float(best.retrieval_score.detach().cpu()):
                best = RetrievalResult(index, atom, result, score)
        if best is None:
            raise RuntimeError("retrieval failed")
        return best

    def consolidate(self, current: AttributedGraph, retrieval: RetrievalResult, posterior_summary: torch.Tensor, execution_utility: float, transfer_complexity: float, config: ConsolidationConfig | None = None) -> str:
        config = config or ConsolidationConfig()
        if float(execution_utility) < float(config.utility_threshold):
            return "skip"
        if float(transfer_complexity) > float(config.complexity_threshold):
            self.add(PriorAtom(posterior_summary.detach(), current.attributes.detach(), current.measure.detach(), float(execution_utility), 0.0, 1))
            self._prune(config.max_atoms)
            return "expand"
        atom = self.atoms[retrieval.atom_index]
        eta = float(config.update_rate)
        reverse = reverse_normalized_coupling(retrieval.fgw.coupling.detach(), atom.measure.detach().to(retrieval.fgw.coupling.device, retrieval.fgw.coupling.dtype))
        projected = reverse @ posterior_summary.detach().to(reverse.device, reverse.dtype) @ reverse.t()
        atom.consensus = (1.0 - eta) * atom.consensus + eta * projected.to(atom.consensus.device, atom.consensus.dtype)
        atom.utility = (1.0 - eta) * atom.utility + eta * float(execution_utility)
        deviation = torch.mean(torch.abs(projected.to(atom.consensus.device, atom.consensus.dtype) - atom.consensus)).item()
        atom.dispersion = (1.0 - eta) * atom.dispersion + eta * float(deviation)
        atom.support_count += 1
        self._prune(config.max_atoms)
        return "update"

    def _prune(self, max_atoms: Optional[int]) -> None:
        if max_atoms is None or len(self.atoms) <= int(max_atoms):
            return
        self.atoms.sort(key=lambda atom: (float(atom.utility) - float(atom.dispersion)), reverse=True)
        del self.atoms[int(max_atoms):]
