"""Config-driven composition of the production design-run optimizer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag_phy.agents import (
    AnthropicLLMClient,
    CritiqueAgent,
    DesignAgent,
    DesignCandidate,
    LLMClient,
)
from rag_phy.config import (
    AppConfig,
    EmbeddingConfig,
    LLMConfig,
    ModelsConfig,
    load_agent_prompts_config,
    load_candidate_search_config,
    load_config,
    load_engine_weight_lookup_config,
    load_knowledge_config,
    load_models_config,
    load_optimization_config,
    load_orchestration_config,
    load_physics_config,
)
from rag_phy.ingestion import Embedder, SentenceTransformerEmbedder
from rag_phy.knowledge import (
    ChromaVectorStore,
    MaterialConstraintStore,
    VectorStore,
)
from rag_phy.optimization import (
    ConfiguredCandidateSampler,
    OptunaOptimizer,
    SourcedEngineWeightEstimator,
)
from rag_phy.optimization.objective import DesignObjective
from rag_phy.orchestration import DesignWorkflow, JsonlTraceSink, WorkflowTraceSink
from rag_phy.physics.models import CycleResult

LLMClientFactory = Callable[[LLMConfig], LLMClient]
EmbedderFactory = Callable[[EmbeddingConfig], Embedder]
VectorStoreFactory = Callable[[Path, str], VectorStore]
TraceSinkFactory = Callable[[Path], WorkflowTraceSink]


def make_optimizer_factory(
    *,
    candidate_search_config_path: str | Path = "config/candidate_search.yaml",
    source_ledger_path: str | Path = "data/curated/sources.md",
    app_config_path: str | Path = "config/app.yaml",
    models_config_path: str | Path = "config/models.yaml",
    knowledge_config_path: str | Path = "config/knowledge.yaml",
    agent_prompts_config_path: str | Path = "config/agent_prompts.yaml",
    physics_config_path: str | Path = "config/physics_bounds.yaml",
    orchestration_config_path: str | Path = "config/orchestration.yaml",
    optimization_config_path: str | Path = "config/optimization.yaml",
    engine_weight_config_path: str | Path = "config/engine_weight.yaml",
    llm_client_factory: LLMClientFactory = AnthropicLLMClient,
    embedder_factory: EmbedderFactory = SentenceTransformerEmbedder,
    vector_store_factory: VectorStoreFactory = ChromaVectorStore,
    trace_sink_factory: TraceSinkFactory = JsonlTraceSink,
) -> Callable[[Path], OptunaOptimizer]:
    """Create per-run optimizers only from typed config and source-ledger-validated inputs.

    Candidate ranges are mandatory; this factory intentionally has no physical-range defaults.
    External model, vector-store, and provider components are injectable for offline tests.
    """

    def create(trace_path: Path) -> OptunaOptimizer:
        app_config: AppConfig = load_config(app_config_path)
        candidate_config = load_candidate_search_config(
            candidate_search_config_path,
            source_ledger_path,
        )
        engine_weight_config = load_engine_weight_lookup_config(
            engine_weight_config_path,
            source_ledger_path,
        )
        models_config: ModelsConfig = load_models_config(models_config_path)
        knowledge_config = load_knowledge_config(knowledge_config_path)
        prompts_config = load_agent_prompts_config(agent_prompts_config_path)
        physics_config = load_physics_config(physics_config_path)
        orchestration_config = load_orchestration_config(orchestration_config_path)
        optimization_config = load_optimization_config(optimization_config_path)

        llm_config = app_config.llm
        if llm_config is None:
            raise ValueError("LLM configuration is required to compose the design workflow")
        if models_config.embedding.provider != "sentence-transformers":
            raise ValueError(
                f"Unsupported embedding provider: {models_config.embedding.provider}"
            )
        if models_config.vector_store.provider != "chroma":
            raise ValueError(f"Unsupported vector-store provider: {models_config.vector_store.provider}")

        llm_client = llm_client_factory(llm_config)
        embedder = embedder_factory(models_config.embedding)
        vector_store = vector_store_factory(
            models_config.vector_store.persist_directory,
            models_config.vector_store.collection_name,
        )
        material_store = MaterialConstraintStore.from_csv(
            knowledge_config.material_constraints_csv_path
        )
        weight_estimator = SourcedEngineWeightEstimator(engine_weight_config)
        objective = DesignObjective(optimization_config.objective)

        def score_candidate(candidate: DesignCandidate, performance: CycleResult) -> float:
            weight_n = weight_estimator(candidate, performance)
            return objective.score_feasible(performance, weight_n)

        workflow = DesignWorkflow(
            design_agent=DesignAgent(llm_client, prompts_config.design_agent),
            physics_config=physics_config,
            critique_agent=CritiqueAgent(
                constraint_store=material_store,
                vector_store=vector_store,
                embedder=embedder,
                explainer=llm_client,
                models_config=models_config,
                config=prompts_config.critique_agent,
            ),
            scorer=score_candidate,
            orchestration_config=orchestration_config,
            trace_sink=trace_sink_factory(trace_path),
        )
        return OptunaOptimizer(
            config=optimization_config,
            workflow=workflow,
            candidate_sampler=ConfiguredCandidateSampler(candidate_config),
            weight_estimator=weight_estimator,
        )

    return create


__all__ = ["make_optimizer_factory"]
