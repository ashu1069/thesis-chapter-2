import os
import argparse
import numpy as np
import pandas as pd

from utils.config import VaccineData


def set_seed(seed: int) -> None:
	"""Set numpy RNG seed for reproducibility."""
	np.random.seed(seed)


def _ar1_series(length: int, phi: float, level: float, noise_std: float) -> np.ndarray:
	"""Generate a bounded AR(1)-like series in [0, 1]."""
	values = np.zeros(length, dtype=np.float32)
	values[0] = np.clip(level, 0.0, 1.0)
	for t in range(1, length):
		innov = np.random.normal(0.0, noise_std)
		values[t] = np.clip(phi * values[t - 1] + (1.0 - phi) * level + innov, 0.0, 1.0)
	return values


def generate_synthetic_time_series(num_steps: int, seed: int = 123, lmic: bool = False):
	"""Generate synthetic static, known, and unknown time series matching VaccineData schema.

	Returns three dataframes: static_df, known_df, unknown_df
	All values are in [0, 1], with temporal correlation for time-based features.
	Targets (four outbreak metrics) are modeled as nonlinear functions of drivers.
	"""
	set_seed(seed)

	static_cols = VaccineData.STATIC_VAR_LIST
	known_cols = VaccineData.TIME_KNOWN_VAR_LIST
	unknown_cols = VaccineData.TIME_UNKNOWN_VAR_LIST

	# Helper: sample R0 by disease category to match literature ranges
	# Returns arrays of R0 and duration (years)
	def sample_r0_and_duration(n: int):
		# Disease categories and R0 ranges (approximate, from user-provided refs)
		diseases = [
			("COVID-19", 1.5, 6.7, (1, 10)),
			("Hepatitis B", 1.3, 2.0, (3, 20)),
			("Hib", 2.5, 4.0, (3, 15)),
			("Yellow Fever", 1.3, 7.2, (1, 10)),
			("Measles", 12.0, 18.0, (5, 25)),
			("Meningococcal", 1.0, 1.7, (1, 5)),
			("Polio", 10.0, 15.0, (3, 15)),
			("Pneumococcal", 1.2, 4.7, (2, 12)),
			("HPV", 1.0, 4.0, (3, 25)),
			("Cholera", 1.0, 3.0, (1, 5)),
			("Japanese Encephalitis", 1.0, 3.0, (1, 8)),
			("Typhoid", 2.0, 7.0, (2, 10)),
			("Malaria", 1.0, 500.0, (5, 30)),  # cap upper tail for stability
		]
		# Weights favor common vaccine-preventable diseases; malaria kept but rare high R0 tail
		weights = np.array([0.15, 0.06, 0.05, 0.06, 0.08, 0.06, 0.06, 0.07, 0.07, 0.07, 0.06, 0.06, 0.05], dtype=np.float64)
		weights = weights / weights.sum()
		choices = np.random.choice(len(diseases), size=n, p=weights)
		R0 = np.zeros(n, dtype=np.float32)
		duration_years = np.zeros(n, dtype=np.float32)
		for i, idx in enumerate(choices):
			name, lo, hi, dur_range = diseases[idx]
			# Use a truncated normal around the mid, clipped to [lo, hi]
			mid = 0.5 * (lo + hi)
			std = 0.15 * (hi - lo)
			val = np.random.normal(mid, std)
			R0[i] = np.clip(val, lo, hi)
			# Duration: uniform in a plausible range (years)
			duration_years[i] = np.random.uniform(dur_range[0], dur_range[1])
		return R0, duration_years

	# Build static features with realistic scales for select columns
	R0_vals, duration_vals = sample_r0_and_duration(num_steps)
	# Population density (people per km^2): log-normal-ish, cap at 2000
	if lmic:
		pop_density = np.clip(np.random.lognormal(mean=np.log(250), sigma=0.9, size=num_steps), 10, 2000).astype(np.float32)
	else:
		pop_density = np.clip(np.random.lognormal(mean=np.log(200), sigma=0.7, size=num_steps), 10, 2000).astype(np.float32)
	# GDP per capita (USD): log-normal, cap at 120k (LMIC skew lower)
	if lmic:
		gdp_pc = np.clip(np.random.lognormal(mean=np.log(6000), sigma=0.9, size=num_steps), 300, 120000).astype(np.float32)
	else:
		gdp_pc = np.clip(np.random.lognormal(mean=np.log(15000), sigma=0.9, size=num_steps), 300, 120000).astype(np.float32)
	# Other static features in [0,1]
	def beta_arr(a, b):
		return np.clip(np.random.beta(a, b, size=num_steps), 0.0, 1.0).astype(np.float32)

	# LMIC skews: higher poverty, lower education/affordability, lower immunization budget later
	if lmic:
		gini = beta_arr(2.5, 2.2)
		poverty = beta_arr(2.8, 2.0)
		afford = beta_arr(2.0, 2.5)
		employment = beta_arr(2.0, 2.5)
		education = beta_arr(2.0, 2.8)
		env_index = beta_arr(2.0, 2.2)
	else:
		gini = beta_arr(2.0, 2.8)
		poverty = beta_arr(2.0, 2.5)
		afford = beta_arr(2.5, 2.0)
		employment = beta_arr(2.5, 2.0)
		education = beta_arr(2.5, 2.0)
		env_index = beta_arr(2.0, 2.0)

	static_df = pd.DataFrame({
		'Endemic_Potential_R0': R0_vals,
		'Endemic_Potential_Duration': duration_vals,  # years
		'Demography_Urban_Rural_Split': beta_arr(2.0, 2.0),
		'Demography_Population_Density': pop_density,
		'Environmental_Index': env_index,
		'Socio_economic_Gini_Index': gini,
		'Socio_economic_Poverty_Rates': poverty,
		'Communication_Affordability': afford,
		'Socio_economic_GDP_per_capita': gdp_pc,
		'Socio_economic_Employment_Rates': beta_arr(2.5, 2.0),
		'Socio_economic_Education_Levels': beta_arr(2.5, 2.0),
	})

	# Known time-dependent features: correlated AR(1) processes
	known_data = {}
	for col in known_cols:
		phi = np.random.uniform(0.6, 0.9)
		# LMIC skew: slightly lower healthcare capacity and budgets, lower stability, lower immunization
		if lmic and col in [
			'Healthcare_Index_Tier_X_hospitals',
			'Healthcare_Index_Workforce_capacity',
			'Healthcare_Index_Bed_availability_per_capita',
			'Healthcare_Index_Expenditure_per_capita',
			'Immunization_Coverage',
			'Economic_Index_Budget_allocation_per_capita',
			'Economic_Index_Fraction_of_total_budget',
			'Political_Stability_Index'
		]:
			level = np.random.uniform(0.2, 0.6)
			noise_std = np.random.uniform(0.03, 0.1)
		else:
			level = np.random.uniform(0.3, 0.7)
			noise_std = np.random.uniform(0.02, 0.08)
		known_data[col] = _ar1_series(num_steps, phi=phi, level=level, noise_std=noise_std)
	known_df = pd.DataFrame(known_data)

	# Build a latent risk driver from a linear mix of known + normalized static, plus smooth noise
	# Use weights to ensure multiple drivers contribute
	w_known = np.random.uniform(-0.6, 0.6, size=len(known_cols)).astype(np.float32)
	# Normalize wide-scale static variables before mixing
	static_for_mix = static_df.copy()
	# Scale population density and GDP to [0,1] by clipping and min-max
	static_for_mix['Demography_Population_Density'] = (static_for_mix['Demography_Population_Density'] / 2000.0).clip(0.0, 1.0)
	static_for_mix['Socio_economic_GDP_per_capita'] = (static_for_mix['Socio_economic_GDP_per_capita'] / 120000.0).clip(0.0, 1.0)
	# Duration: cap at 30 years and scale
	static_for_mix['Endemic_Potential_Duration'] = (static_for_mix['Endemic_Potential_Duration'] / 30.0).clip(0.0, 1.0)
	# R0: cap at 20 for mixing scale (keep raw value in CSV)
	static_for_mix['Endemic_Potential_R0'] = (static_for_mix['Endemic_Potential_R0'] / 20.0).clip(0.0, 1.0)

	w_static = np.random.uniform(-0.4, 0.4, size=len(static_cols)).astype(np.float32)
	# Compute a smooth driver over time using a few key known drivers
	known_matrix = known_df.values  # [T, K]
	# Create a slow-moving shock component
	shock = _ar1_series(num_steps, phi=0.95, level=0.5, noise_std=0.02)
	# Use per-row normalized static vectors to influence risk
	linear_driver = (known_matrix @ w_known) * 0.4 + (static_for_mix.values @ w_static) * 0.3 + shock * 0.3
	linear_driver = np.clip(linear_driver, -2.0, 2.0)

	# Unknown time-dependent features include the four targets + Security_and_Conflict_Index input
	# Map to [0,1] using tanh nonlinearity with small independent noise per series
	noise = lambda s: np.random.normal(0.0, s, size=num_steps).astype(np.float32)
	Frequency_of_outbreaks = np.clip(0.5 + 0.6 * np.tanh(linear_driver + noise(0.08)), 0.0, 1.0)
	Magnitude_of_outbreaks_Deaths = np.clip(0.5 + 0.6 * np.tanh(0.9 * linear_driver + noise(0.08)), 0.0, 1.0)
	Magnitude_of_outbreaks_Infected = np.clip(0.5 + 0.6 * np.tanh(1.1 * linear_driver + 0.1 + noise(0.08)), 0.0, 1.0)
	Magnitude_of_outbreaks_Severity_Index = np.clip(0.5 + 0.6 * np.tanh(linear_driver + noise(0.08)), 0.0, 1.0)
	Security_and_Conflict_Index = _ar1_series(num_steps, phi=0.85, level=np.random.uniform(0.3, 0.7), noise_std=0.05)

	unknown_df = pd.DataFrame({
		'Frequency_of_outbreaks': Frequency_of_outbreaks,
		'Magnitude_of_outbreaks_Deaths': Magnitude_of_outbreaks_Deaths,
		'Magnitude_of_outbreaks_Infected': Magnitude_of_outbreaks_Infected,
		'Magnitude_of_outbreaks_Severity_Index': Magnitude_of_outbreaks_Severity_Index,
		'Security_and_Conflict_Index': Security_and_Conflict_Index,
	})

	# Ensure column order matches config exactly
	unknown_df = unknown_df[unknown_cols]

	return static_df, known_df, unknown_df


def main():
	parser = argparse.ArgumentParser(description="Generate synthetic dataset matching VaccineData schema.")
	parser.add_argument("--steps", type=int, default=10000, help="Number of time steps to generate")
	parser.add_argument("--seed", type=int, default=123, help="Random seed")
	parser.add_argument("--lmic", action='store_true', help="Skew feature distributions toward LMIC profiles")
	parser.add_argument("--outdir", type=str, default=None, help="Output directory for CSVs (defaults to chapter-2/data)")
	args = parser.parse_args()

	default_outdir = os.path.join(os.path.dirname(__file__), "..", "data")
	outdir = os.path.abspath(args.outdir) if args.outdir else os.path.abspath(default_outdir)
	os.makedirs(outdir, exist_ok=True)

	static_df, known_df, unknown_df = generate_synthetic_time_series(num_steps=args.steps, seed=args.seed, lmic=args.lmic)

	static_path = os.path.join(outdir, "static_data_synthetic.csv")
	known_path = os.path.join(outdir, "time_dependent_known_data_synthetic.csv")
	unknown_path = os.path.join(outdir, "time_dependent_unknown_data_synthetic.csv")

	static_df.to_csv(static_path, index=False)
	known_df.to_csv(known_path, index=False)
	unknown_df.to_csv(unknown_path, index=False)

	print("Synthetic data written:")
	print(static_path)
	print(known_path)
	print(unknown_path)


if __name__ == "__main__":
	main()


