"""Typed inputs and station results for the single-spool turbojet cycle."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CycleInput(BaseModel):
    """Turbojet design point; temperatures are K, pressures Pa, speeds m/s, and flow kg/s."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ambient_temperature_k: float = Field(gt=0)
    ambient_pressure_pa: float = Field(gt=0)
    flight_speed_m_per_s: float = Field(ge=0)
    air_mass_flow_kg_per_s: float = Field(gt=0)
    compressor_pressure_ratio: float = Field(gt=1)
    turbine_inlet_temperature_k: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_turbine_temperature(self) -> CycleInput:
        """Require a turbine inlet temperature above the static inlet temperature."""
        if self.turbine_inlet_temperature_k <= self.ambient_temperature_k:
            raise ValueError("Turbine inlet temperature must exceed ambient temperature")
        return self


@dataclass(frozen=True)
class StationState:
    """Total station state; temperature K, pressure Pa, enthalpy J/kg, entropy J/(kg K)."""

    station: str
    total_temperature_k: float
    total_pressure_pa: float
    enthalpy_j_per_kg: float
    entropy_j_per_kg_k: float
    fuel_air_ratio: float = 0.0
    exit_velocity_m_per_s: float | None = None
    exit_area_m2: float | None = None
    static_pressure_pa: float | None = None
    static_temperature_k: float | None = None
    static_enthalpy_j_per_kg: float | None = None


@dataclass(frozen=True)
class CycleResult:
    """Turbojet performance; thrust N, SFC kg/(N s), and efficiencies dimensionless."""

    thrust_n: float
    specific_fuel_consumption_kg_per_n_s: float
    thermal_efficiency: float
    fuel_air_ratio: float
    stations: tuple[StationState, ...]


__all__ = ["CycleInput", "CycleResult", "StationState"]
