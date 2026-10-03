# Known Assumptions and Bounds

This register distinguishes cited model-envelope/worked-example values from universal
engineering limits. No value here is a certification or safe-operating envelope.

## `config/physics_bounds.yaml`

| Setting | Config value | Evidence status | Evidence / limitation |
| --- | --- | --- | --- |
| `working_fluid` | Air | model-choice | Model choice. Combustion products are approximated as Air; this is not an equilibrium combustion model. |
| `fuel_lower_heating_value_j_per_kg` | 43000000 | source-backed | NASA/CR-20210000284, p. 37 (printed p. 40), uses 43 MJ/kg as ideal jet-fuel/kerosene specific energy and states that combustion fuels use LHV. Representative input, not a Jet-A specification or a particular engine fuel measurement. [NASA report](https://ntrs.nasa.gov/api/citations/20210000284/downloads/1502_Datta__CR%2020210000284_081821.pdf) |
| `temperature_min_k` | 60 | source-backed | Rounded upward from the 59.75 K lower property range. EOS envelope, not a cycle design bound. [Lemmon et al. (2000)](https://doi.org/10.1063/1.1285884) |
| `temperature_max_k` | 2000 | source-backed | Upper Air EOS range; at high temperatures/pressures, the source notes reliance on nitrogen data because direct Air data are absent. Not a cycle design bound. [Lemmon et al. (2000)](https://doi.org/10.1063/1.1285884) |
| `pressure_max_pa` | 2000000000 | source-backed | Upper Air EOS pressure range. It is not an allowable engine pressure. [Lemmon et al. (2000)](https://doi.org/10.1063/1.1285884) |
| `combustor_pressure_loss_fraction` | 0.04 | source-backed | IIT Bombay NPTEL Lecture 11, Problem 1. Representative worked-example input, not a universal combustor loss. [NPTEL lecture](https://archive.nptel.ac.in/content/storage2/courses/101101002/downloads/Lect-11.pdf) |
| `component_efficiencies.intake` | 0.93 | source-backed | NPTEL Lecture 11, Problem 1 benchmark; one specified worked-example input, not a sourced distribution or hardware guarantee. [NPTEL lecture](https://archive.nptel.ac.in/content/storage2/courses/101101002/downloads/Lect-11.pdf) |
| `component_efficiencies.compressor` | 0.87 | source-backed | NPTEL Lecture 11, Problem 1 benchmark; one specified worked-example input, not a sourced distribution or hardware guarantee. [NPTEL lecture](https://archive.nptel.ac.in/content/storage2/courses/101101002/downloads/Lect-11.pdf) |
| `component_efficiencies.combustor` | 0.98 | source-backed | NPTEL Lecture 11, Problem 1 benchmark; one specified worked-example input, not a sourced distribution or hardware guarantee. [NPTEL lecture](https://archive.nptel.ac.in/content/storage2/courses/101101002/downloads/Lect-11.pdf) |
| `component_efficiencies.turbine` | 0.9 | source-backed | NPTEL Lecture 11, Problem 1 benchmark; one specified worked-example input, not a sourced distribution or hardware guarantee. [NPTEL lecture](https://archive.nptel.ac.in/content/storage2/courses/101101002/downloads/Lect-11.pdf) |
| `component_efficiencies.mechanical` | 0.99 | source-backed | NPTEL Lecture 11, Problem 1 benchmark; one specified worked-example input, not a sourced distribution or hardware guarantee. [NPTEL lecture](https://archive.nptel.ac.in/content/storage2/courses/101101002/downloads/Lect-11.pdf) |
| `component_efficiencies.nozzle` | 0.95 | source-backed | NPTEL Lecture 11, Problem 1 benchmark; one specified worked-example input, not a sourced distribution or hardware guarantee. [NPTEL lecture](https://archive.nptel.ac.in/content/storage2/courses/101101002/downloads/Lect-11.pdf) |
| `numerics.root_pressure_tolerance_pa` | 0.1 | numerical-policy | Numerical solver tolerance selected by the implementation; no external source establishes this as an engineering requirement. Requires numerical convergence and sensitivity review. |
| `numerics.root_max_iterations` | 128 | numerical-policy | Numerical iteration cap selected by the implementation; no external source establishes this as an engineering requirement. Requires convergence testing for new operating domains. |

The NPTEL inputs are useful regression/benchmark conditions, but the model currently
uses them as configurable defaults. A real candidate sampler must not treat a single
worked example as a verified feasible design space. The Phase 8 cycle-bound verification
gate therefore remains open pending reviewed operating-domain sources.

Run `python scripts/validate_physics_bounds.py` to check that every leaf in the physics
configuration has exactly one assumptions-register entry, that the register's displayed
value matches the config, and that every entry has an allowed evidence disposition. This
structural check does not establish that a cited source is technically adequate; engineering
review is still required for the cycle-design domain.

## Material Constraints

See `data/curated/sources.md`. In particular, the SiC/SiC CMC row applies to NASA's
specific System A, not generic CMCs, and the Inconel 718 values are a historical
alloy/application limit. The current critique compares turbine-inlet gas temperature directly with these material
screens. It does not calculate blade metal temperature or cooling effectiveness, and may
reject feasible cooled designs. The duplicate bare/cooled Inconel entries intentionally
use one material-level temperature value; they do not model cooling.

SC 180 now has a manufacturer-sourced screening value in the curated table. The stated
1000 C property-temperature claim is not a qualified allowable or validated service limit;
see `SEAH-SC180` in the source ledger. CMSX-4 itself still has no general service-temperature
row. Do not optimize or claim a material-feasible design based on this incomplete table or
mistake the SC 180 screen for component qualification.
