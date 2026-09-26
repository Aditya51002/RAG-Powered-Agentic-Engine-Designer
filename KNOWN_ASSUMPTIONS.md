# Known Assumptions and Bounds

This register distinguishes cited model-envelope/worked-example values from universal
engineering limits. No value here is a certification or safe-operating envelope.

## `config/physics_bounds.yaml`

| Setting | Value | Evidence and limitation |
| --- | --- | --- |
| `working_fluid` | Air | Model choice. Combustion products are approximated as Air; this is not an equilibrium combustion model. |
| `fuel_lower_heating_value_j_per_kg` | 43,000,000 J/kg | **Unverified placeholder.** The previous comment attributed it to an NPTEL Lecture 34 problem, but its linked source page has not been confirmed to contain this value. Fuel identity/composition is also not separately modeled. Do not treat this as a verified engineering input. |
| `temperature_min_k` | 60 K | Rounded upward from the 59.75 K lower range reported by Lemmon et al. (2000), DOI 10.1063/1.1285884. This is a property-model envelope, not a cycle design bound. |
| `temperature_max_k` | 2000 K | Upper temperature range reported by Lemmon et al. (2000), DOI 10.1063/1.1285884. At high temperatures/pressures the paper notes property predictions rely on nitrogen data because direct air data are absent. |
| `pressure_max_pa` | 2,000,000,000 Pa | Upper pressure range reported by the same Air equation-of-state source. It is not an allowable engine pressure. |
| `combustor_pressure_loss_fraction` | 0.04 | IIT Bombay NPTEL Lecture 11, Problem 1 benchmark. A representative worked-example input, not a universal combustor loss. |
| intake/compressor/combustor/turbine/mechanical/nozzle efficiencies | 0.93/0.87/0.98/0.90/0.99/0.95 | IIT Bombay NPTEL Lecture 11, Problem 1 benchmark. These are one problem's specified component efficiencies, not sourced distributions or guaranteed hardware performance. |
| `numerics.root_pressure_tolerance_pa` | 0.1 Pa | Numerical solver tolerance selected by the implementation; no external source establishes it as an engineering requirement. Requires numerical convergence/sensitivity review. |
| `numerics.root_max_iterations` | 128 | Numerical iteration cap selected by the implementation. It is not a physical bound and requires convergence testing for new operating domains. |

The NPTEL inputs are useful regression/benchmark conditions, but the model currently
uses them as configurable defaults. A real candidate sampler must not treat a single
worked example as a verified feasible design space. The Phase 8 cycle-bound verification
gate therefore remains open pending reviewed operating-domain sources.

## Material Constraints

See `data/curated/sources.md`. In particular, the SiC/SiC CMC row applies to NASA's
specific System A, not generic CMCs, and the Inconel 718 values are a historical
alloy/application limit. The current critique compares turbine-inlet gas temperature directly with these material
screens. It does not calculate blade metal temperature or cooling effectiveness, and may
reject feasible cooled designs. The duplicate bare/cooled Inconel entries intentionally
use one material-level temperature value; they do not model cooling.

CMSX-4/single-crystal superalloy coverage has no validated service-temperature row yet.
Do not optimize or claim a material-feasible design based on this incomplete table.
