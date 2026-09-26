# Curated Data Source Ledger

This ledger is the provenance key for `material_constraints.csv`. The numeric kelvin
values are unit conversions from the cited source values; they are not additional test data.

## NASA-CR-19690012409

- NASA Contractor Report, *An Advanced Small Gas Turbine Engine for Aircraft Auxiliary Power Units*, report 19690012409.
- Official full text: <https://ntrs.nasa.gov/api/citations/19690012409/downloads/19690012409.pdf>
- Relevant statement: the report identifies PWA 1010 (Inconel 718), notes good strength up to 1300 deg F, and discusses an engine turbine inlet temperature of 1225 deg F. 1300 deg F converts to 977.594 K.
- Scope: this is a historical alloy/application statement, not a modern design allowable. The `bare` and `cooled` names are two application aliases for the same alloy-level metal-temperature screen. Cooling hardware does not change the material value; this application currently has no blade-metal model.

## CARPENTER-TI6AL4V

- Carpenter Technology, Ti 6Al-4V alloy finder, manufacturer recommendation.
- Official page: <https://www.carpentertechnology.com/alloy-finder/Ti-6Al-4V>
- Relevant statement: service temperatures up to approximately 660 deg F (350 deg C). 350 deg C converts to 623.15 K.
- Scope: manufacturer recommendation for the alloy generally; it does not qualify a specific product form, stress, environment, or life.

## NASA-TM-2006-20060054003

- NASA Technical Memorandum 2006-20060054003, *Advanced SiC/SiC Ceramic Composite Systems Developed for High-Temperature Structural Applications*.
- Official full text: <https://ntrs.nasa.gov/api/citations/20060054003/downloads/20060054003.pdf>
- Relevant evidence: NASA reports upper-use temperature for the specific SiC/SiC systems A and B at 1450 deg C. 1450 deg C converts to 1723.15 K.
- Scope: this is a specific NASA system-level upper-use value, not a safe maximum for all CMCs or an allowable design limit. Do not generalize it beyond the named system or ignore stress, exposure duration, environment, coating, and life requirements.

## Coverage Not Yet Resolved

CMSX-4 or another single-crystal superalloy has not been assigned a maximum service temperature. Public creep-rupture test points are condition-dependent and do not by themselves establish a general service limit. Do not infer an allowable temperature from a test point. The requested superalloy coverage remains an open Phase 8 item until a suitably qualified manufacturer or handbook limit is reviewed.
