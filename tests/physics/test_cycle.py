from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from rag_phy.config import PhysicsConfig, load_physics_config
from rag_phy.physics.cycle import CycleDomainError, simulate_cycle
from rag_phy.physics.models import CycleInput


@pytest.fixture
def physics_config() -> PhysicsConfig:
    """Return the sourced, typed NPTEL real-cycle configuration."""
    config_path = Path(__file__).parents[2] / "config" / "physics_bounds.yaml"
    return load_physics_config(config_path)


def make_nptel_problem_one_inputs(turbine_inlet_temperature_k: float = 1200) -> CycleInput:
    """Build NPTEL Lecture 11 Problem 1 conditions (K, Pa, m/s, kg/s)."""
    return CycleInput(
        ambient_temperature_k=223.3,
        ambient_pressure_pa=26_500,
        flight_speed_m_per_s=239.6,
        air_mass_flow_kg_per_s=1,
        compressor_pressure_ratio=8,
        turbine_inlet_temperature_k=turbine_inlet_temperature_k,
    )


def test_nptel_worked_turbojet_example(physics_config: PhysicsConfig) -> None:
    """Compare EOS station and performance values with the published worked example.

    The lecture uses constant specific heats; CoolProp's real-air EOS causes small,
    expected deviations in performance totals.
    """
    result = simulate_cycle(make_nptel_problem_one_inputs(), physics_config)
    station_by_name = {station.station: station for station in result.stations}

    assert station_by_name["2"].total_temperature_k == pytest.approx(251.9, abs=0.1)
    assert station_by_name["3"].total_temperature_k == pytest.approx(486.8, abs=2)
    assert station_by_name["5"].total_temperature_k == pytest.approx(992.3, abs=10)
    assert station_by_name["7"].static_temperature_k == pytest.approx(850.7, abs=5)
    assert station_by_name["7"].static_pressure_pa == pytest.approx(67_100, rel=0.04)
    assert result.thrust_n == pytest.approx(596.25, rel=0.03)
    assert result.specific_fuel_consumption_kg_per_n_s == pytest.approx(3.32e-5, rel=0.07)


@settings(max_examples=15, derandomize=True)
@given(
    temperatures=st.tuples(
        st.floats(min_value=900, max_value=1500, allow_nan=False, allow_infinity=False),
        st.floats(min_value=900, max_value=1500, allow_nan=False, allow_infinity=False),
    ).filter(lambda pair: pair[0] < pair[1])
)
def test_higher_turbine_inlet_temperature_does_not_reduce_thrust(
    temperatures: tuple[float, float],
) -> None:
    """Increasing TIT over the test range must not reduce net thrust, all else fixed."""
    config_path = Path(__file__).parents[2] / "config" / "physics_bounds.yaml"
    physics_config = load_physics_config(config_path)
    low_result = simulate_cycle(
        make_nptel_problem_one_inputs(temperatures[0]), physics_config
    )
    high_result = simulate_cycle(
        make_nptel_problem_one_inputs(temperatures[1]), physics_config
    )

    assert high_result.thrust_n >= low_result.thrust_n


@pytest.mark.parametrize(
    "overrides",
    [
        {"compressor_pressure_ratio": -1},
        {"compressor_pressure_ratio": 1},
        {"turbine_inlet_temperature_k": 200},
    ],
)
def test_invalid_inputs_raise_validation_errors(overrides: dict[str, float]) -> None:
    """Reject nonphysical pressure ratios and turbine temperatures before simulation."""
    fields = {
        "ambient_temperature_k": 223.3,
        "ambient_pressure_pa": 26_500,
        "flight_speed_m_per_s": 239.6,
        "air_mass_flow_kg_per_s": 1,
        "compressor_pressure_ratio": 8,
        "turbine_inlet_temperature_k": 1200,
    }
    fields.update(overrides)

    with pytest.raises(ValidationError):
        CycleInput.model_validate(fields)


def test_turbine_temperature_beyond_eos_range_is_rejected(
    physics_config: PhysicsConfig,
) -> None:
    """Do not pass temperatures beyond the sourced CoolProp Air EOS range."""
    inputs = make_nptel_problem_one_inputs(2050)

    with pytest.raises(CycleDomainError, match="EOS limits"):
        simulate_cycle(inputs, physics_config)
