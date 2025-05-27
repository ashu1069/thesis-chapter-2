import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from data import BaseEM, CFR, R0

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    method_name: str
    true_params: Dict[str, float]
    estimated_params: Dict[str, float]
    convergence: bool
    iterations: int
    log_likelihood: float
    computation_time: float
    metrics: Dict[str, float]

class DataGenerator:
    """Generate synthetic epidemiological data for benchmarking."""
    
    @staticmethod
    def generate_cfr_data(
        n_cases: int,
        true_cfr: float,
        missing_ratio: float = 0.2,
        noise_level: float = 0.05
    ) -> Dict[str, int]:
        """
        Generate synthetic CFR data.
        
        Args:
            n_cases: Total number of cases
            true_cfr: True case fatality rate
            missing_ratio: Proportion of cases with unknown outcomes
            noise_level: Level of noise to add to the data
        """
        # Generate true outcomes
        true_deaths = int(n_cases * true_cfr)
        true_recoveries = n_cases - true_deaths
        
        # Add noise
        noise = np.random.normal(0, noise_level * n_cases, 1)[0]
        noisy_deaths = max(0, min(n_cases, int(true_deaths + noise)))
        noisy_recoveries = n_cases - noisy_deaths
        
        # Create missing data
        n_missing = int(n_cases * missing_ratio)
        missing_deaths = int(n_missing * true_cfr)
        missing_recoveries = n_missing - missing_deaths
        
        return {
            'n': n_cases,
            'd': noisy_deaths - missing_deaths,
            'u': n_missing
        }

    @staticmethod
    def generate_r0_data(
        n_timepoints: int,
        true_r0: float,
        gamma: float,
        initial_cases: int = 10,
        noise_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Generate synthetic R0 data using a simple SIR model.
        
        Args:
            n_timepoints: Number of time points
            true_r0: True R0 value
            gamma: Recovery rate
            initial_cases: Initial number of cases
            noise_level: Level of noise to add
        """
        I = [initial_cases]
        for t in range(1, n_timepoints):
            new_cases = int(I[-1] * true_r0 * (1 - gamma))
            I.append(new_cases)
        
        # Add noise
        I = np.array(I)
        noise = np.random.normal(0, noise_level * I, len(I))
        I = np.maximum(0, I + noise).astype(int)
        
        return {
            'I': I,
            'gamma': gamma
        }

class BenchmarkMetrics:
    """Calculate various metrics for comparing estimation methods."""
    
    @staticmethod
    def calculate_metrics(
        true_params: Dict[str, float],
        estimated_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate various error metrics."""
        metrics = {}
        
        for param_name in true_params:
            true_val = true_params[param_name]
            est_val = estimated_params[param_name]
            
            # Absolute error
            metrics[f'{param_name}_abs_error'] = abs(true_val - est_val)
            
            # Relative error
            if true_val != 0:
                metrics[f'{param_name}_rel_error'] = abs(true_val - est_val) / abs(true_val)
            
            # Percentage error
            metrics[f'{param_name}_pct_error'] = abs(true_val - est_val) / abs(true_val) * 100
        
        return metrics

class BenchmarkRunner:
    """Run benchmarks comparing different estimation methods."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(
        self,
        data_generator: Callable,
        estimation_methods: List[Callable],
        true_params: Dict[str, float],
        n_runs: int = 100,
        **data_kwargs
    ) -> List[BenchmarkResult]:
        """
        Run benchmark comparing multiple estimation methods.
        
        Args:
            data_generator: Function to generate synthetic data
            estimation_methods: List of estimation method classes
            true_params: Dictionary of true parameter values
            n_runs: Number of benchmark runs
            **data_kwargs: Additional arguments for data generation
        """
        for _ in range(n_runs):
            # Generate synthetic data
            data = data_generator(**data_kwargs)
            
            # Run each estimation method
            for method_class in estimation_methods:
                import time
                start_time = time.time()
                
                # Initialize and run the method
                method = method_class(**data)
                result = method.fit()
                
                computation_time = time.time() - start_time
                
                # Calculate metrics
                metrics = BenchmarkMetrics.calculate_metrics(
                    true_params,
                    result['final_params']
                )
                
                # Store results
                benchmark_result = BenchmarkResult(
                    method_name=method_class.__name__,
                    true_params=true_params,
                    estimated_params=result['final_params'],
                    convergence=result['converged'],
                    iterations=result['iterations'],
                    log_likelihood=result['final_log_likelihood'],
                    computation_time=computation_time,
                    metrics=metrics
                )
                
                self.results.append(benchmark_result)
        
        return self.results
    
    def plot_results(self, metric_name: str):
        """Plot benchmark results for a specific metric."""
        # Convert results to DataFrame
        df = pd.DataFrame([
            {
                'method': r.method_name,
                'value': r.metrics[metric_name],
                'converged': r.convergence
            }
            for r in self.results
        ])
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='method', y='value')
        plt.title(f'Distribution of {metric_name} by Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> pd.DataFrame:
        """Generate a summary report of benchmark results."""
        summary_data = []
        
        for result in self.results:
            summary_data.append({
                'method': result.method_name,
                'convergence_rate': result.convergence,
                'avg_iterations': result.iterations,
                'avg_computation_time': result.computation_time,
                **{f'avg_{k}': v for k, v in result.metrics.items()}
            })
        
        return pd.DataFrame(summary_data).groupby('method').mean()

# Example usage
if __name__ == "__main__":
    # Define benchmark parameters
    true_cfr = 0.05
    n_cases = 1000
    missing_ratio = 0.2
    
    # Initialize benchmark runner
    runner = BenchmarkRunner()
    
    # Run CFR benchmark
    cfr_results = runner.run_benchmark(
        data_generator=DataGenerator.generate_cfr_data,
        estimation_methods=[CFR],
        true_params={'cfr': true_cfr},
        n_cases=n_cases,
        true_cfr=true_cfr,
        missing_ratio=missing_ratio
    )
    
    # Plot results
    runner.plot_results('cfr_rel_error')
    
    # Generate report
    report = runner.generate_report()
    print("\nBenchmark Report:")
    print(report) 