"""Stable public schemas for design-run and knowledge-search HTTP endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DesignRunRequest(APIModel):
    design_goal: str = Field(min_length=1)
    trial_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_goal_not_blank(self) -> DesignRunRequest:
        if not self.design_goal.strip():
            raise ValueError("design_goal must not be blank")
        return self


class CandidateResponse(APIModel):
    ambient_temperature_k: float
    ambient_pressure_pa: float
    flight_speed_m_per_s: float
    air_mass_flow_kg_per_s: float
    compressor_pressure_ratio: float
    turbine_inlet_temperature_k: float
    hot_section_material_name: str


class PerformanceResponse(APIModel):
    thrust_n: float
    specific_fuel_consumption_kg_per_n_s: float
    thermal_efficiency: float
    fuel_air_ratio: float


class ParetoDesignResponse(APIModel):
    candidate: CandidateResponse
    thrust_to_weight_ratio: float
    specific_fuel_consumption_kg_per_n_s: float
    cited_sources: tuple[str, ...]


class DesignRunResultResponse(APIModel):
    accepted_candidate: CandidateResponse | None
    performance: PerformanceResponse | None
    pareto_frontier: tuple[ParetoDesignResponse, ...]
    cited_sources: tuple[str, ...]


class DesignProgressResponse(APIModel):
    iteration: int = Field(gt=0)
    total_iterations: int = Field(gt=0)
    status: str
    best_candidate: CandidateResponse
    best_score: float
    best_valid: bool


class DesignRunResponse(APIModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: datetime
    updated_at: datetime
    progress: DesignProgressResponse | None = None
    result: DesignRunResultResponse | None = None
    error: str | None = None


class DesignRunSubmissionResponse(APIModel):
    run_id: str
    status: Literal["queued"]


class TraceEventResponse(APIModel):
    trace_id: str
    span_id: str
    parent_span_id: str | None
    node: str
    iteration: int
    message: str
    timestamp: str


class KnowledgeResultResponse(APIModel):
    text: str
    source_refs: tuple[str, ...]
    distance: float


class HealthResponse(APIModel):
    status: Literal["ready", "not_ready"]
