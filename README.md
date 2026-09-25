# RAG Phy

A grounded, software-only decision-support prototype for preliminary jet-engine design-space exploration. It does not produce CFD/FEA results or certify a flight-worthy design.

## Current Phase

Phase 0 provides the package skeleton, typed YAML/environment configuration, and shared structured logging. Phase 1 adds a deterministic single-spool turbojet cycle using CoolProp properties. Phase 2 adds document ingestion, dense retrieval, and a source-required material constraint store. Phase 3 adds an offline-testable, schema-validated design proposal agent. Phase 4 adds code-determined, citation-required design critique. Phase 5 wires these components into a bounded LangGraph workflow. Phase 6 adds a configurable Optuna objective and incremental Pareto frontier.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,knowledge,optimization]'
pytest
```

Install `.[embeddings]` as well to run the configured BGE model locally. The first real-model run downloads the configured model weights.

## Configuration

The initial application configuration is in `config/app.yaml`. `rag_phy.config.load_config` is the only module that reads YAML or process environment variables. Environment overrides use the `RAG_PHY_` prefix and `__` for nested keys; for example, `RAG_PHY_LOGGING__LEVEL=DEBUG`.

`models.yaml` configures chunk windows, BGE, and Chroma. `knowledge.yaml` points to the curated material constraint CSV. `physics_bounds.yaml` holds sourced cycle values, `agent_prompts.yaml` holds agent prompts and retry/history controls, `orchestration.yaml` configures graph iteration limits and convergence tolerance, and `optimization.yaml` configures Optuna and objective parameters. Typed loaders live in `rag_phy.config`.

## Design Proposal Agent

`DesignAgent` accepts an injected `LLMClient`; it has no provider-specific API dependency. It validates responses against the typed `DesignCandidate` schema and retries malformed or previously rejected parameter sets up to `design_agent.max_attempts`. Prompt text, retry limit, rejection-history bound, and signature precision are configured in `agent_prompts.yaml` and loaded with `load_agent_prompts_config`. Candidate/history rows in tests are synthetic, not design recommendations. The proposal agent does not decide constraint validity; that is handled by `CritiqueAgent` below.

`CritiqueAgent` compares the candidate's turbine inlet temperature (K) with the selected hot-section material's curated maximum service temperature (K) in code, retrieves literature passages, and requires their source references before returning `valid=True`. The explanation model can contextualize evidence but cannot determine the verdict. **Screening limitation:** this phase compares turbine inlet gas temperature with a material service rating because the physics model has no blade-metal thermal/cooling model. This is not a metal-temperature prediction and cannot establish thermal safety; it may reject feasible cooled turbine designs. Synthetic test rows are not engineering limits, and no curated production corpus/table is present in this workspace; real evidence is required for a valid production critique.

## Orchestration

`DesignWorkflow` compiles explicit `propose -> simulate -> critique -> revise|score -> optimizer_step -> continue|converge` nodes. Invoke it with a goal; inject the proposal, critique, simulator, and finite scalar scorer. Larger scores are treated as better. `orchestration.yaml` bounds total proposals and invalid revisions and supplies score-space convergence tolerance. The `optimizer_step` tracks best-so-far and convergence; the Phase 6 Optuna runner can seed one candidate per graph invocation. The returned state includes typed rejection/trial histories and per-node transition events. Tests inject all components and use no live model calls.

## Optimization

`DesignObjective` scores valid candidates as `w_ttw * (T/W / T/W_reference) - w_sfc * (SFC / SFC_reference)`. Invalid candidates subtract `base_penalty + severity_weight * normalized_violation_severity`, preserving a graded penalty for numeric constraint exceedance rather than mapping every rejection to zero. `ParetoFrontier` incrementally tracks non-dominated valid candidates by maximizing T/W and minimizing SFC. `OptunaOptimizer` runs the actual Optuna study through Phase 5 in single-candidate mode and records citations and metrics on each trial.

The optimizer requires injected `CandidateSampler` and `EngineWeightEstimator` implementations. No verified search ranges or engine weight model were supplied, so this repository deliberately does not invent them; the integration test uses synthetic ranges, performance, and weight only. The config's equal objective weights and penalty coefficients are initial policy assumptions to calibrate against the mission. Its SFC normalization reference is the NPTEL worked-cycle regression value, not a requirement or production target. A real study needs verified parameter bounds, a defensible weight estimator, and curated evidence.

## Physics Model

Load the typed settings using `load_physics_config("config/physics_bounds.yaml")`, create a `CycleInput`, and call `simulate_cycle(inputs, config)`. Input and output field names carry SI units. The model assumes a single-spool turbojet, represents combustion products as CoolProp Air, neglects fuel sensible enthalpy, applies configured station efficiencies, and models a convergent nozzle with real-fluid sonic choking and pressure thrust. These are preliminary cycle estimates, not flight or manufacturing validation. The fuel value, limits, and benchmark efficiencies in `physics_bounds.yaml` have citations in its comments.

## Knowledge Pipeline

`IngestionPipeline` composes `DocumentLoader`, `TextChunker`, an injected `Embedder`, and `VectorStore`. `SentenceTransformerEmbedder` loads the configured BGE model on first use; `ChromaVectorStore` persists vectors under its configured path. PDF text extraction uses PyMuPDF; scanned/image-only PDFs fail with an explicit empty-text error because OCR is not part of this phase.

`MaterialConstraintStore.from_csv` expects `material_name,max_service_temperature_k,source_id`. Exact material access is indexed by normalized name, and temperature range queries use a sorted index. Missing materials and missing source IDs raise errors. No curated source corpus or material table was present in the workspace, so `data/curated/material_constraints.csv` must be populated from verified records before the production store can be used. The rows under `tests/fixtures` are synthetic test data and must not be used as engineering limits.
