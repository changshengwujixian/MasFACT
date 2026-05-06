# MasFACT

This repository contains the official implementation of the paper submitted to NeurIPS 2026:

**\textsc{MasFACT}: Continual Multi-Agent Topology Learning via Geometry-Aware Posterior Transfer**

MasFACT is a framework for continual topology learning in LLM-based multi-agent systems. It aims to mitigate topology forgetting, where adapting a multi-agent topology generator to new tasks may overwrite previously effective collaboration structures. The framework preserves reusable historical topology priors, aligns them to incoming tasks through geometry-aware matching, and performs conservative posterior adaptation to balance new-task plasticity and old-task stability.

## Anonymous Review Notice

This repository is prepared for anonymous peer review. All author-identifying information has been removed to comply with the double-blind review policy. The repository is intended solely to help reviewers inspect the implementation details and reproduce the main experimental results reported in the paper.

## Conceptual Scope

\textsc{MasFACT} studies continual learning at the level of MAS communication topology. When tasks arrive sequentially, the central question is not only whether model parameters can be preserved, but whether reusable collaboration structures can be retained, aligned, and adapted without collapsing into task-specific topology search.

The implementation reflects this view through three principles.

1. **Historical topology knowledge is represented explicitly.** High-utility communication structures are stored as prior atoms containing structural, semantic, and mass components.
2. **Transfer is geometry-aware.** Prior atoms are aligned to a new task through Fused Gromov-Wasserstein transport, so that topology reuse is not tied to fixed agent indices.
3. **Adaptation is conservative.** New-task topology posteriors are formed as sparse residual deviations around aligned priors, with complexity controlled by the PAC-Bayes-inspired objective used in the paper.

## Repository Structure

```text
MasFACT_core/
├── masfact/
│   ├── types.py
│   ├── fgw.py
│   ├── posterior.py
│   ├── objectives.py
│   ├── memory.py
│   ├── adapters.py
│   └── masfact.py
├── protocols/
│   └── build_continual_mas_protocol.py
├── pyproject.toml
└── README.md
```

## Core Method Files

`types.py` defines the mathematical data structures used by the method, including attributed task graphs, historical prior atoms, topology posteriors, and adaptation outputs.

`fgw.py` implements the geometry-aware alignment layer. It builds semantic and relational cost objects, solves a softened FGW alignment problem, and projects a selected historical consensus topology into the current task space.

`posterior.py` implements the stochastic topology posterior used for conservative adaptation. It represents task-specific topology edits as residual changes around an aligned structural prior while enforcing feasible communication masks.

`objectives.py` implements the paper's risk-control quantities, including the PAC-Bayes transfer bound surrogate, posterior-prior complexity, and sparse residual regularization.

`memory.py` implements factorized topology prior storage, retrieval, and selective consolidation. Historical atoms are updated only when the adapted topology is both useful and structurally reliable.

`adapters.py` defines a minimal plug-and-play interface for connecting \textsc{MasFACT} to different MAS topology optimizers. The interface treats an external optimizer as a source of task-side scaffold graphs, while \textsc{MasFACT} handles memory retrieval, alignment, and conservative adaptation.

`masfact.py` assembles these components into the high-level retrieve--align--adapt--consolidate procedure described in the methodology.

## Continual MAS Protocol Construction

The `protocols/` directory contains a dataset construction utility for the hierarchical continual MAS evaluation protocol. It downloads selected HuggingFace datasets, normalizes record fields, and reorganizes them into the same `CL-dataset`-style structure used in the paper.

The protocol code documents four levels of evaluation design:

1. task-level shifts across reasoning families;
2. domain-level shifts within knowledge QA, mathematics, code generation, and multi-hop QA;
3. coarse class-level shifts inside major benchmarks;
4. fine-grained class-level shifts for long-horizon continual evaluation.

This part of the repository is included because the benchmark protocol is itself part of the paper's contribution: it makes topology-level continual adaptation measurable across heterogeneous MAS collaboration pressures. The protocol builder only constructs dataset files and metadata. It does not perform model training, model inference, reward computation, or experimental evaluation.