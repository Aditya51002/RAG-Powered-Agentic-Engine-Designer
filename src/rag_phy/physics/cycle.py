"""Pure station-by-station single-spool turbojet cycle calculation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

from rag_phy.config import PhysicsConfig
from rag_phy.physics.models import CycleInput, CycleResult, StationState
from rag_phy.physics.properties import CoolPropBackend, PropertyBackend


class CycleDomainError(ValueError):
    """Raised when a design point cannot represent a valid cycle state."""


@dataclass(frozen=True)
class _Context:
    """Immutable calculation dependencies shared by station transformations."""

    inputs: CycleInput
    config: PhysicsConfig
    properties: PropertyBackend

    def prop(self, output: str, key_a: str, value_a: float, key_b: str, value_b: float) -> float:
        """Evaluate a configured working-fluid property and enforce finite SI results."""
        result = self.properties.evaluate(
            output, key_a, value_a, key_b, value_b, self.config.working_fluid
        )
        if not isfinite(result):
            raise CycleDomainError(f"Property {output} returned a non-finite value")
        return result


def _station(
    context: _Context,
    name: str,
    temperature_k: float,
    pressure_pa: float,
    fuel_air_ratio: float = 0.0,
    exit_velocity_m_per_s: float | None = None,
    exit_area_m2: float | None = None,
    static_pressure_pa: float | None = None,
    static_temperature_k: float | None = None,
    static_enthalpy_j_per_kg: float | None = None,
) -> StationState:
    """Construct a state from total temperature and pressure using the configured EOS."""
    if not context.config.temperature_min_k <= temperature_k <= context.config.temperature_max_k:
        raise CycleDomainError(f"Station {name} temperature is outside configured EOS limits")
    if not 0 < pressure_pa <= context.config.pressure_max_pa:
        raise CycleDomainError(f"Station {name} pressure is outside configured EOS limits")
    enthalpy = context.prop("H", "T", temperature_k, "P", pressure_pa)
    entropy = context.prop("S", "T", temperature_k, "P", pressure_pa)
    return StationState(
        station=name,
        total_temperature_k=temperature_k,
        total_pressure_pa=pressure_pa,
        enthalpy_j_per_kg=enthalpy,
        entropy_j_per_kg_k=entropy,
        fuel_air_ratio=fuel_air_ratio,
        exit_velocity_m_per_s=exit_velocity_m_per_s,
        exit_area_m2=exit_area_m2,
        static_pressure_pa=static_pressure_pa,
        static_temperature_k=static_temperature_k,
        static_enthalpy_j_per_kg=static_enthalpy_j_per_kg,
    )


def inlet_station(context: _Context) -> StationState:
    """Diffuse the flight stream; temperatures K, pressure Pa, speed m/s."""
    inputs = context.inputs
    ambient_h = context.prop(
        "H", "T", inputs.ambient_temperature_k, "P", inputs.ambient_pressure_pa
    )
    ambient_s = context.prop(
        "S", "T", inputs.ambient_temperature_k, "P", inputs.ambient_pressure_pa
    )
    total_h = ambient_h + inputs.flight_speed_m_per_s**2 / 2
    ideal_pressure = context.prop("P", "H", total_h, "S", ambient_s)
    intake_efficiency = context.config.component_efficiencies.intake
    total_pressure = inputs.ambient_pressure_pa + intake_efficiency * (
        ideal_pressure - inputs.ambient_pressure_pa
    )
    total_temperature = context.prop("T", "H", total_h, "P", total_pressure)
    return _station(context, "2", total_temperature, total_pressure)


def compressor_station(context: _Context, inlet: StationState) -> StationState:
    """Compress the inlet total state to the candidate pressure ratio."""
    outlet_pressure = inlet.total_pressure_pa * context.inputs.compressor_pressure_ratio
    ideal_enthalpy = context.prop(
        "H", "P", outlet_pressure, "S", inlet.entropy_j_per_kg_k
    )
    efficiency = context.config.component_efficiencies.compressor
    outlet_enthalpy = inlet.enthalpy_j_per_kg + (
        ideal_enthalpy - inlet.enthalpy_j_per_kg
    ) / efficiency
    outlet_temperature = context.prop("T", "P", outlet_pressure, "H", outlet_enthalpy)
    return _station(context, "3", outlet_temperature, outlet_pressure)


def combustor_station(context: _Context, compressor_exit: StationState) -> StationState:
    """Add heat at the candidate turbine inlet temperature and calculate fuel ratio."""
    pressure = compressor_exit.total_pressure_pa * (
        1 - context.config.combustor_pressure_loss_fraction
    )
    temperature = context.inputs.turbine_inlet_temperature_k
    burner_enthalpy = context.prop("H", "T", temperature, "P", pressure)
    enthalpy_rise = burner_enthalpy - compressor_exit.enthalpy_j_per_kg
    # Fuel sensible enthalpy is omitted; the energy balance uses the compressor-exit
    # enthalpy as its zero reference for both streams.
    heat_available = (
        context.config.component_efficiencies.combustor
        * context.config.fuel_lower_heating_value_j_per_kg
        - enthalpy_rise
    )
    if heat_available <= 0 or enthalpy_rise <= 0:
        raise CycleDomainError("Combustor energy balance does not yield a positive fuel ratio")
    fuel_air_ratio = enthalpy_rise / heat_available
    state = _station(context, "4", temperature, pressure, fuel_air_ratio)
    return state


def turbine_station(
    context: _Context,
    inlet: StationState,
    compressor_exit: StationState,
    combustor_exit: StationState,
) -> StationState:
    """Expand hot gas by the shaft work needed to drive the compressor."""
    gas_mass_ratio = 1 + combustor_exit.fuel_air_ratio
    shaft_efficiency = context.config.component_efficiencies.mechanical
    compressor_work = compressor_exit.enthalpy_j_per_kg - inlet.enthalpy_j_per_kg
    actual_enthalpy_drop = compressor_work / (gas_mass_ratio * shaft_efficiency)
    isentropic_enthalpy_drop = actual_enthalpy_drop / context.config.component_efficiencies.turbine
    isentropic_exit_enthalpy = combustor_exit.enthalpy_j_per_kg - isentropic_enthalpy_drop
    if isentropic_exit_enthalpy <= 0 or actual_enthalpy_drop <= 0:
        raise CycleDomainError("Turbine work balance is outside the valid enthalpy range")
    exit_pressure = context.prop(
        "P", "H", isentropic_exit_enthalpy, "S", combustor_exit.entropy_j_per_kg_k
    )
    actual_exit_enthalpy = combustor_exit.enthalpy_j_per_kg - actual_enthalpy_drop
    exit_temperature = context.prop("T", "P", exit_pressure, "H", actual_exit_enthalpy)
    return _station(
        context, "5", exit_temperature, exit_pressure, combustor_exit.fuel_air_ratio
    )


def _nozzle_mach(context: _Context, turbine_exit: StationState, pressure_pa: float) -> float:
    """Compute nozzle Mach number at a pressure using real-fluid static properties."""
    ideal_enthalpy = context.prop(
        "H", "P", pressure_pa, "S", turbine_exit.entropy_j_per_kg_k
    )
    enthalpy_drop = turbine_exit.enthalpy_j_per_kg - ideal_enthalpy
    if enthalpy_drop <= 0:
        return 0.0
    velocity = sqrt(2 * context.config.component_efficiencies.nozzle * enthalpy_drop)
    actual_enthalpy = turbine_exit.enthalpy_j_per_kg - velocity**2 / 2
    sound_speed = context.prop("A", "P", pressure_pa, "H", actual_enthalpy)
    return velocity / sound_speed


def nozzle_station(context: _Context, turbine_exit: StationState) -> StationState:
    """Expand through a convergent nozzle, solving its real-fluid sonic pressure."""
    back_pressure = context.inputs.ambient_pressure_pa
    if turbine_exit.total_pressure_pa <= back_pressure:
        raise CycleDomainError("Turbine exit pressure must exceed ambient back pressure")

    if _nozzle_mach(context, turbine_exit, back_pressure) >= 1:
        low_pressure = back_pressure
        high_pressure = turbine_exit.total_pressure_pa
        for _ in range(context.config.numerics.root_max_iterations):
            if high_pressure - low_pressure <= context.config.numerics.root_pressure_tolerance_pa:
                break
            midpoint = (low_pressure + high_pressure) / 2
            if _nozzle_mach(context, turbine_exit, midpoint) >= 1:
                low_pressure = midpoint
            else:
                high_pressure = midpoint
        if high_pressure - low_pressure > context.config.numerics.root_pressure_tolerance_pa:
            raise CycleDomainError("Nozzle sonic-pressure solver did not converge")
        exit_pressure = (low_pressure + high_pressure) / 2
    else:
        exit_pressure = back_pressure

    ideal_exit_enthalpy = context.prop(
        "H", "P", exit_pressure, "S", turbine_exit.entropy_j_per_kg_k
    )
    velocity = sqrt(
        2
        * context.config.component_efficiencies.nozzle
        * (turbine_exit.enthalpy_j_per_kg - ideal_exit_enthalpy)
    )
    actual_exit_enthalpy = turbine_exit.enthalpy_j_per_kg - velocity**2 / 2
    exit_temperature = context.prop("T", "P", exit_pressure, "H", actual_exit_enthalpy)
    density = context.prop("D", "P", exit_pressure, "H", actual_exit_enthalpy)
    exhaust_mass_flow = context.inputs.air_mass_flow_kg_per_s * (
        1 + turbine_exit.fuel_air_ratio
    )
    area = exhaust_mass_flow / (density * velocity)
    return _station(
        context,
        "7",
        turbine_exit.total_temperature_k,
        turbine_exit.total_pressure_pa,
        turbine_exit.fuel_air_ratio,
        velocity,
        area,
        exit_pressure,
        exit_temperature,
        actual_exit_enthalpy,
    )


def simulate_cycle(
    inputs: CycleInput,
    config: PhysicsConfig,
    properties: PropertyBackend | None = None,
) -> CycleResult:
    """Run the deterministic turbojet station pipeline and calculate performance.

    The fuel-product stream is approximated as air because a sourced combustion-product
    composition is outside this phase. Pressure thrust is included for choked operation.
    """
    if not config.temperature_min_k <= inputs.ambient_temperature_k <= config.temperature_max_k:
        raise CycleDomainError("Ambient temperature is outside configured EOS limits")
    if inputs.ambient_pressure_pa > config.pressure_max_pa:
        raise CycleDomainError("Ambient pressure exceeds configured EOS limits")
    if inputs.turbine_inlet_temperature_k > config.temperature_max_k:
        raise CycleDomainError("Turbine inlet temperature exceeds configured EOS limits")

    context = _Context(inputs, config, properties or CoolPropBackend())
    inlet = inlet_station(context)
    compressor_exit = compressor_station(context, inlet)
    combustor_exit = combustor_station(context, compressor_exit)
    turbine_exit = turbine_station(context, inlet, compressor_exit, combustor_exit)
    nozzle_exit = nozzle_station(context, turbine_exit)

    assert nozzle_exit.exit_velocity_m_per_s is not None
    assert nozzle_exit.exit_area_m2 is not None
    exhaust_velocity = nozzle_exit.exit_velocity_m_per_s
    exhaust_mass_flow = inputs.air_mass_flow_kg_per_s * (1 + combustor_exit.fuel_air_ratio)
    thrust = (
        exhaust_mass_flow * exhaust_velocity
        - inputs.air_mass_flow_kg_per_s * inputs.flight_speed_m_per_s
        + (nozzle_exit.static_pressure_pa - inputs.ambient_pressure_pa)
        * nozzle_exit.exit_area_m2
    )
    fuel_flow = inputs.air_mass_flow_kg_per_s * combustor_exit.fuel_air_ratio
    if thrust <= 0:
        raise CycleDomainError("Calculated net thrust is not positive")
    specific_fuel_consumption = fuel_flow / thrust
    jet_power_per_air_mass = (
        exhaust_mass_flow * exhaust_velocity**2
        - inputs.air_mass_flow_kg_per_s * inputs.flight_speed_m_per_s**2
    ) / (2 * inputs.air_mass_flow_kg_per_s)
    thermal_efficiency = jet_power_per_air_mass / (
        combustor_exit.fuel_air_ratio * config.fuel_lower_heating_value_j_per_kg
    )
    return CycleResult(
        thrust_n=thrust,
        specific_fuel_consumption_kg_per_n_s=specific_fuel_consumption,
        thermal_efficiency=thermal_efficiency,
        fuel_air_ratio=combustor_exit.fuel_air_ratio,
        stations=(inlet, compressor_exit, combustor_exit, turbine_exit, nozzle_exit),
    )


__all__ = [
    "CycleDomainError",
    "combustor_station",
    "compressor_station",
    "inlet_station",
    "nozzle_station",
    "simulate_cycle",
    "turbine_station",
]
