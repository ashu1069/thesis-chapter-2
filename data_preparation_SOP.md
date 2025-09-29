## Standard Operating Procedure (SOP): Data Preparation and Simulation for TFT

### 1) Scope and Goal
- **Goal**: Prepare consistent, large-scale time series datasets for training the Temporal Fusion Transformer (TFT) on multi-objective vaccine allocation tasks.
- **Outputs**: Three CSV files with aligned time steps and feature schemas:
  - `static_data_*.csv`
  - `time_dependent_known_data_*.csv`
  - `time_dependent_unknown_data_*.csv`

### 2) Environment and Reproducibility
- Python 3.10+ and packages from `requirements.txt`.
- Reproducibility: the generator sets `numpy` seed; pass `--seed` to control.
- Project root for module resolution when running scripts:
```bash
export PYTHONPATH=/home/stu12/s11/ak1825/thesis/thesis-chapter-2
```

### 3) Data Sources and Provenance
- Real-world anchor for ranges/scales:
  - R0 ranges per disease (COVID-19, HepB, Hib, Yellow Fever, Measles, Meningococcal, Polio, Pneumococcal, HPV, Cholera, Japanese Encephalitis, Typhoid, Malaria) compiled from cited literature in the thesis.
  - Example COVID static record (`data/static_data_COVID-19.csv`) informs units: `Endemic_Potential_Duration` in years, GDP per capita in USD, etc.
- Synthetic generator encodes these priors and produces diverse, LMIC-skewable datasets.

### 4) Files and Schemas
- Static CSV columns (per time row, used as per-row country profile):
  - `Endemic_Potential_R0` (float, disease-informed ranges)
  - `Endemic_Potential_Duration` (float, years)
  - `Demography_Urban_Rural_Split` [0,1]
  - `Demography_Population_Density` (people per km^2)
  - `Environmental_Index` [0,1]
  - `Socio_economic_Gini_Index` [0,1]
  - `Socio_economic_Poverty_Rates` [0,1]
  - `Communication_Affordability` [0,1]
  - `Socio_economic_GDP_per_capita` (USD)
  - `Socio_economic_Employment_Rates` [0,1]
  - `Socio_economic_Education_Levels` [0,1]

- Time-known CSV columns (all in [0,1] unless otherwise noted):
  - `Healthcare_Index_Tier_X_hospitals`
  - `Healthcare_Index_Workforce_capacity`
  - `Healthcare_Index_Bed_availability_per_capita`
  - `Healthcare_Index_Expenditure_per_capita`
  - `Immunization_Coverage`
  - `Economic_Index_Budget_allocation_per_capita`
  - `Economic_Index_Fraction_of_total_budget`
  - `Political_Stability_Index`

- Time-unknown CSV columns (targets included):
  - `Frequency_of_outbreaks` (TARGET)
  - `Magnitude_of_outbreaks_Deaths` (TARGET)
  - `Magnitude_of_outbreaks_Infected` (TARGET)
  - `Magnitude_of_outbreaks_Severity_Index` (TARGET)
  - `Security_and_Conflict_Index`

Notes:
- The training loader expects windows of length 5 and uses the last step as targets.
- Normalization is performed inside the dataset class; raw CSVs keep natural units where applicable.

### 5) Generation Logic (experiments/generate_synthetic_dataset.py)
- Static features:
  - `R0` sampled from a mixture over disease-specific ranges reflecting literature; optional LMIC mode does not alter the ranges but affects co-features.
  - `Duration` sampled in years by disease; stored raw.
  - `Population_Density`: log-normal (capped at 2000 per km^2).
  - `GDP_per_capita`: log-normal (capped at 120k USD).
  - Remaining indices sampled in [0,1] with Beta distributions; LMIC mode skews toward higher poverty/gini and lower education/affordability.
- Time-known features:
  - Generated via bounded AR(1)-like processes to induce temporal autocorrelation.
  - LMIC mode shifts means lower for healthcare capacity/expenditure, immunization, budgets, and stability, with slightly higher noise.
- Time-unknown features and targets:
  - Latent risk driver is a linear mix of time-known features and a normalized static vector plus a smooth shock.
  - Targets apply tanh transforms with small noise to yield bounded signals in [0,1].

### 6) Commands
- Generate 10k steps (generic):
```bash
PYTHONPATH=$PYTHONPATH python3 experiments/generate_synthetic_dataset.py \
  --steps 10000 --seed 123 --outdir ./data
```
- Generate 10k steps with LMIC profile:
```bash
PYTHONPATH=$PYTHONPATH python3 experiments/generate_synthetic_dataset.py \
  --steps 10000 --seed 123 --lmic --outdir ./data
```
- Train TFT on generated data:
```bash
STATIC_CSV=./data/static_data_synthetic.csv \
KNOWN_CSV=./data/time_dependent_known_data_synthetic.csv \
UNKNOWN_CSV=./data/time_dependent_unknown_data_synthetic.csv \
PYTHONPATH=$PYTHONPATH python3 train_full_features.py | cat
```

### 7) Validation Checklist
- File presence and column order:
  - All three CSVs exist and exactly match the expected column names above.
- Row alignment:
  - `len(static) == len(known) == len(unknown) == STEPS`.
- Basic stats (spot check):
  - `R0` distribution spans expected disease ranges; tail events (e.g., measles, polio) present.
  - `Duration` is positive and realistic in years.
  - LMIC run: median GDPpc lower than generic; poverty/gini higher; immunization and healthcare indices lower.
- Loader sanity:
  - Dataset length ≥ 5; no NaNs after normalization; one training step runs without error.

### 8) Configuration Options
- `--steps`: number of time steps (rows) to generate.
- `--seed`: RNG seed.
- `--lmic`: skew distributions to LMIC profiles (static + time-known).
- Output directory: use `--outdir` to select a target folder.

### 9) Troubleshooting
- `ModuleNotFoundError: utils`: ensure `PYTHONPATH` points to `thesis/thesis-chapter-2`.
- Exploding loss or NaNs: lower `--steps` for quick tests; verify CSV column order and no extra headers/blank lines.
- Unbalanced disease mix: adjust mixture weights in the generator or switch to uniform sampling if needed.

### 10) Extending/Customizing
- Disease mix: parameterize disease weights or provide a YAML to fix per-disease proportions.
- Country labels: add a `Country_ID`/`Disease_Label` column in static CSV for stratified sampling.
- Window size: update dataset window length if training requires longer sequences.

### 11) Governance
- Keep track of generator version and seed in experiment logs.
- Store generated CSVs with a date/seed suffix for provenance when used in results.


