# Curated Data Source Ledger

This ledger is the provenance key for `material_constraints.csv`. The numeric kelvin
values are unit conversions from the cited source values; they are not additional test data.
It also records source IDs used by optimizer inputs under `config/`.

The document used for the draft QA corpus is independently identified and checksummed in
`corpus/manifest.json`; the manifest source IDs use repository-relative PDF paths and page numbers.

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

## SEAH-SC180

- SeAH Superalloy Technologies, SC 180 cast alloy product page: <https://www.seahsuperalloys.com/superalloys/sc-180>
- The manufacturer identifies SC 180 as a single-crystal nickel-base alloy used for high-pressure turbine blades and describes yield, tensile, and creep-rupture properties at temperatures up to 1000 deg C. `material_constraints.csv` converts that stated temperature to 1273.15 K.
- Scope: this is a manufacturer-stated temperature for alloy properties, used as a preliminary material screening threshold. It is not a component allowable, safe service limit, life-qualified design value, or substitute for stress-, orientation-, coating-, environment-, and duration-specific data. The current model also compares gas temperature rather than predicted blade-metal temperature.

## NASA-TM-2004-213048-CORPUS

- J.A. DiCarlo et al., *SiC/SiC Composites for 1200 C and Above*, NASA/TM-2004-213048 (2004).
- Official full text: <https://ntrs.nasa.gov/api/citations/20040191405/downloads/20040191405.pdf>
- Local corpus document: `data/curated/corpus/NASA-TM-2004-213048.pdf`; its SHA-256 and corpus version are in `data/curated/corpus/manifest.json`.
- Scope: indexed for the draft evaluation set only. It is not a CMSX-4 source and its CMC observations do not imply material limits for other systems.

## Coverage Not Yet Resolved

CMSX-4 or another single-crystal superalloy has not been assigned a maximum service temperature. Public creep-rupture test points are condition-dependent and do not by themselves establish a general service limit. Do not infer an allowable temperature from a test point. The requested superalloy coverage remains an open Phase 8 item until a suitably qualified manufacturer or handbook limit is reviewed.

## NASA-TM-X-73199-TURBOJET-DATA

- M.H. Waters and E.T. Schairer, *Analysis of Turbofan Propulsion System Weight and Dimensions*, NASA-TM-X-73199 (1977), Table 1, printed p. 29, `Turbojets` subsection.
- Official full text: <https://ntrs.nasa.gov/api/citations/19770012125/downloads/19770012125.pdf>
- The table reports sea-level-static (SLS) thrust and dry mass for CJ805-3 (49,817 N; 1,270 kg), CJ610-1 (12,677 N; 181 kg), CJ610-8 (13,789 N; 185 kg), JT4A-3 (70,278 N; 2,277 kg), and JT3C-6 (60,048 N; 1,920 kg). These are the lookup anchors in `config/engine_weight.yaml`; only directly printed N and kg values are used.
- Scope: historical turbojet hardware data, not a design correlation or a guarantee for a newly designed engine. The implementation linearly interpolates dry mass against candidate net thrust only between the lowest and highest listed SLS thrusts, converts mass to force using the standard-gravity convention below, and refuses extrapolation. Candidate thrust is not necessarily SLS-rated thrust, and interpolation across different engine families is a preliminary approximation requiring validation for a selected engine class.

## NASA-CR-20170000884-J85-MODEL

- J. Csank, *Practical Techniques for Modeling Gas Turbine Engine Performance*, NASA/CR-2017-000884 (2017), J85 turbojet example.
- Official full text: <https://ntrs.nasa.gov/api/citations/20170000884/downloads/20170000884.pdf>
- Table 1 reports the modeled J85 takeoff reference point: sea level, standard day, Mach 0, 2,850 lbf net thrust, 0.99 (lbm/hr)/lbf SFC, 16,540 rpm, 44 lbm/s air flow, compressor pressure ratio 7, and turbine inlet temperature 2,100 R. The variant is not named in that table. The paper reports its own calibrated T-MATS model matching the targets within 1%; that result applies to the paper's model, not this repository's cycle model.
- Appendix Table A1 gives component settings/assumptions for that model, including compressor efficiency 0.87, burner LHV 18,400 BTU/lbm, turbine efficiency 0.85, and 0.01 pps cooling flow. The report states component values were iteratively tuned and assumes shaft-speed-limited operation.
- Scope: educational model/design-point reference, not a certified engine operating envelope or a set of independent optimizer bounds. The compressor-map analysis has no actual-map data below 80% corrected speed, extends map regions using generic maps, and uses a generic turbine map because an actual turbine map was not found. The paper's flight envelope is based on a Viper Jet use case, and it omits rarely used corners. The repository comparison dataset records the published point in SI using the separately cited NIST conversion factors; use it to diagnose model discrepancy, not to imply calibration or off-design validity.

## NIST-SP811-UNIT-CONVERSIONS

- NIST, *Guide to the SI*, Special Publication 811, Appendix B conversion factors: <https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors>.
- Used for conversion of the NASA J85 point from lbm, lbf, psi, and Rankine to SI units in `data/curated/reference_engines/j85_takeoff.json`. The stored converted digits are arithmetic representations of the rounded source quantities, not additional measurement precision.
- Scope: unit conversion only; it provides no engine-performance evidence or operating limits.

## NIST-SP-811-STANDARD-GRAVITY

- NIST, *Guide to the SI*, Appendix B.8, standard acceleration of free fall `g_n = 9.80665 m/s^2` (exact conventional value).
- Official source: <https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b8>
- Used only to convert the NASA table's dry mass in kg to force in N for the existing thrust-to-weight objective. It is not an engine-design parameter.
