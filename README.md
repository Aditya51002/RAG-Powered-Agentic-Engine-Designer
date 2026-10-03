# RAG Phy

A grounded, software-only decision-support prototype for preliminary jet-engine design-space exploration. It does not produce CFD/FEA results or certify a flight-worthy design.

## Current Phase

Phases 0–7 provide the configuration, cycle model, retrieval, agents, workflow, optimization interfaces, and evaluation framework. Phase 8 has a draft NASA corpus, draft QA labels, curated material values, and a bounds-assumption ledger; it is **not complete** because independent QA review, full evaluation with a generation/evaluation backend, and qualified single-crystal material coverage are still gates. Phase 9 now includes a source-validated sampler interface and a preliminary dry-mass lookup based on five historical NASA turbojet data points, converted to weight with NIST standard gravity. It remains **incomplete**: verified project search ranges are not yet configured, and interpolation across unlike historical engines with cycle thrust treated as rated SLS thrust needs engineering review; no real optimizer study is justified. Phase 10 adds a configurable Anthropic adapter; a live provider run has not been performed. Phase 11 adds an injected FastAPI surface and fake-backed tests; an optimizer executor adapter is now available, but the default service still has no deployment composition. Phase 12 adds an HTTP-only Streamlit client; neither is deployable end to end until genuine runtime adapters and readiness checks are supplied. Later phases remain gated by their preceding exit criteria and external credentials/infrastructure.

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

`app.yaml` holds the Anthropic model, timeout/retry policy, and configurable per-token pricing used for estimated call-cost logs. The API key is not stored there: provide `RAG_PHY_LLM__API_KEY` in the process environment. Install `.[llm]` to use `AnthropicLLMClient`; its transient transport retry budget is separate from each agent's malformed-response attempts. The adapter logs token counts, elapsed time, estimated cost, and provider request ID without recording prompts or completions. No live call is part of the test suite.

## Design Proposal Agent

`DesignAgent` accepts an injected `LLMClient`; it has no provider-specific API dependency. It validates responses against the typed `DesignCandidate` schema and retries malformed or previously rejected parameter sets up to `design_agent.max_attempts`. Prompt text, retry limit, rejection-history bound, and signature precision are configured in `agent_prompts.yaml` and loaded with `load_agent_prompts_config`. Candidate/history rows in tests are synthetic, not design recommendations. The proposal agent does not decide constraint validity; that is handled by `CritiqueAgent` below.

`CritiqueAgent` compares the candidate's turbine inlet temperature (K) with the selected hot-section material's curated maximum service temperature (K) in code, retrieves literature passages, and requires their source references before returning `valid=True`. The explanation model can contextualize evidence but cannot determine the verdict. **Screening limitation:** this phase compares turbine inlet gas temperature with a material service rating because the physics model has no blade-metal thermal/cooling model. This is not a metal-temperature prediction and cannot establish thermal safety; it may reject feasible cooled turbine designs. Synthetic test rows are not engineering limits, and no curated production corpus/table is present in this workspace; real evidence is required for a valid production critique.

## Orchestration

`DesignWorkflow` compiles explicit `propose -> simulate -> critique -> revise|score -> optimizer_step -> continue|converge` nodes. Invoke it with a goal; inject the proposal, critique, simulator, and finite scalar scorer. Larger scores are treated as better. `orchestration.yaml` bounds total proposals and invalid revisions and supplies score-space convergence tolerance. The `optimizer_step` tracks best-so-far and convergence; the Phase 6 Optuna runner can seed one candidate per graph invocation. The returned state includes typed rejection/trial histories and per-node transition events. Tests inject all components and use no live model calls.

## Optimization

`DesignObjective` scores valid candidates as `w_ttw * (T/W / T/W_reference) - w_sfc * (SFC / SFC_reference)`. Invalid candidates subtract `base_penalty + severity_weight * normalized_violation_severity`, preserving a graded penalty for numeric constraint exceedance rather than mapping every rejection to zero. `ParetoFrontier` incrementally tracks non-dominated valid candidates by maximizing T/W and minimizing SFC. `OptunaOptimizer` runs the actual Optuna study through Phase 5 in single-candidate mode and records citations and metrics on each trial.

`OptunaOptimizer` accepts injected `CandidateSampler` and `EngineWeightEstimator` implementations. `ConfiguredCandidateSampler` consumes `CandidateSearchConfig`, requiring a source ID for every parameter range/material and a source-ledger heading through `load_candidate_search_config`; configured uniform or log-uniform distributions are used as provided. `SourcedEngineWeightEstimator` reads `config/engine_weight.yaml`, interpolates dry mass between source-ledger-backed NASA sea-level-static thrust anchors, converts mass to weight using the configured NIST standard gravity, and refuses extrapolation. The anchors are the historical turbojets tabulated in NASA-TM-X-73199; treating current cycle thrust as comparable to rated SLS thrust and linearly interpolating across unlike engines are explicit preliminary assumptions, not structural mass analysis or a validated correlation. Both require review for the target engine class. No production search bounds are supplied, so no real study is defensible; integration tests use synthetic inputs, while a separate test loads and checks the real sourced anchor dataset. The config's equal objective weights and penalty coefficients are initial policy assumptions to calibrate against the mission. Its SFC normalization reference is the NPTEL worked-cycle regression value, not a requirement or production target.

## Evaluation and Tracing

`config/evaluation.yaml` selects `data/evaluation/qa_set.json`, RAGAS metrics, report location, and a JSONL trace path. The 18-case NASA QA set is explicitly marked draft and its labels need independent review. `scripts/validate_qa_dataset.py` verifies the corpus checksum and page references; `scripts/build_curated_index.py` builds the local BGE/Chroma index; `scripts/evaluate_retrieval.py` runs retrieval-only source-ID precision/recall and records `data/evaluation/retrieval_draft.json`. That retrieval report is a draft diagnostic, not a complete answer-quality/RAGAS evaluation or a production accuracy claim. See `data/evaluation/README.md` for the labeling contract. `evaluate_dataset` requires a target adapter that returns both its answer and the exact retrieved source IDs/passages; it computes macro source-level precision/recall and delegates answer/context scoring to `ConfiguredRagasBackend`. Install `.[evaluation]`, then inject a RAGAS-compatible evaluator LLM (and embeddings when required) to run those metrics.

`cumulative_validity_rate` and `validity_observations_from_study` convert completed optimizer outcomes into cumulative rates; `save_validity_chart` writes a PNG. Pass `JsonlTraceSink(load_evaluation_config(...).trace_jsonl_path)` to `DesignWorkflow` to persist one correlated span event for each graph transition. The sink interface can also be implemented by a hosted tracing backend such as Langfuse; local JSONL tracing needs no credentials. Evaluation tests use explicitly synthetic examples and are contract tests, not claims about retrieval performance or engineering validity.

## Dashboard

Install `.[app]` and run `streamlit run src/rag_phy/app/streamlit_app.py` for the existing in-process prototype dashboard. Configure `dashboard.service_factory` in `config/app.yaml` as `python.module:function`; the factory can return `OptimizationDashboardService` around a configured `OptunaOptimizer`. The UI owns no physics, proposal, retrieval, or scoring logic. The factory remains unset because the source-complete search space and engine-weight data are missing, and no Anthropic API key is configured.

## HTTP API

Install `.[api]` and run `uvicorn rag_phy.api.main:app`. `create_app` exposes asynchronous design-run submission/polling, per-run JSONL trace retrieval, injected knowledge search, and readiness endpoints using explicit public schemas. Each request returns an `X-Request-ID`; it is propagated through structured logs and passed to the run executor for use as the workflow trace ID. The run store is in-memory and loses records on process restart. The checked-in `main:app` has no deployment adapters configured, so readiness is false and design/search requests return structured 503 errors until deployment code constructs the app with a real run executor, retriever, and readiness probe that loads config and checks vector-store reachability. Use `create_app` with fakes for zero-cost API tests; do not treat the default entry point as a completed deployable backend.

Install `.[app]` and run `streamlit run src/rag_phy/frontend/streamlit_app.py` for the API-only browser surface. It accepts an API URL, submits goals/trial counts, polls run state, displays the Pareto set and citation trail, and provides direct knowledge search. Its HTTP client module imports no server-side domain models. This UI cannot produce a completed design until the API is constructed with the deployment-owned adapters described above.

## Physics Model

Load the typed settings using `load_physics_config("config/physics_bounds.yaml")`, create a `CycleInput`, and call `simulate_cycle(inputs, config)`. Input and output field names carry SI units. The model assumes a single-spool turbojet, represents combustion products as CoolProp Air, neglects fuel sensible enthalpy, applies configured station efficiencies, and models a convergent nozzle with real-fluid sonic choking and pressure thrust. These are preliminary cycle estimates, not flight or manufacturing validation. The fuel value, limits, and benchmark efficiencies in `physics_bounds.yaml` have citations in its comments.

## Knowledge Pipeline

`IngestionPipeline` composes `DocumentLoader`, `TextChunker`, an injected `Embedder`, and `VectorStore`. `SentenceTransformerEmbedder` loads the configured BGE model on first use; `ChromaVectorStore` persists vectors under its configured path. PDF text extraction uses PyMuPDF; scanned/image-only PDFs fail with an explicit empty-text error because OCR is not part of this phase.

`MaterialConstraintStore.from_csv` expects `material_name,max_service_temperature_k,source_id`. Exact material access is indexed by normalized name, and temperature range queries use a sorted index. Curated values and scope are documented in [the source ledger](data/curated/sources.md); run `python scripts/validate_curated_data.py` to check source IDs. These are screening inputs, not certified material allowables. The single-crystal superalloy coverage and verified cycle-domain evidence remain incomplete; see [KNOWN_ASSUMPTIONS.md](KNOWN_ASSUMPTIONS.md). Rows under `tests/fixtures` are synthetic and must not be used as engineering limits.

The draft evaluation corpus and QA cases are checksummed and page-addressable. Run `python scripts/validate_qa_dataset.py` to verify labels against the source PDF, then `python scripts/build_curated_index.py` to build the configured BGE/Chroma index from the corpus manifest. Index data is generated under `data/chroma/` and is not versioned; the source PDF and SHA-256 manifest are versioned. QA labels still require independent review, and the production RAGAS report is not available until a configured evaluator is supplied.
