import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseEM(ABC):
    """Base class for Expectation-Maximization algorithm in epidemiological data."""
    
    def __init__(self, data: Dict[str, Any], params: Dict[str, Any]):
        """
        Initialize the EM algorithm.
        
        Args:
            data: Dictionary containing the epidemiological data
            params: Dictionary containing the model parameters
        """
        self.data = data
        self.params = params
        self.history = {
            'iterations': [],
            'log_likelihoods': [],
            'parameter_values': []
        }

    @abstractmethod
    def e_step(self) -> Any:
        """Expectation step - compute expected values of missing data."""
        pass

    @abstractmethod
    def m_step(self, expected_values: Any) -> Dict[str, Any]:
        """Maximization step - update parameter estimates."""
        pass

    @abstractmethod
    def log_likelihood(self) -> float:
        """Compute the log-likelihood of the current parameter estimates."""
        pass

    def fit(self, tol: float = 1e-6, max_iter: int = 100) -> Dict[str, Any]:
        """
        Fit the model using the EM algorithm.
        
        Args:
            tol: Convergence tolerance
            max_iter: Maximum number of iterations
            
        Returns:
            Dictionary containing the final parameter estimates and convergence information
        """
        log_likelihood_old = self.log_likelihood()
        
        for i in range(max_iter):
            # E-step
            expected_values = self.e_step()
            
            # M-step
            new_params = self.m_step(expected_values)
            self.params.update(new_params)
            
            # Compute new log-likelihood
            log_likelihood_new = self.log_likelihood()
            
            # Store history
            self.history['iterations'].append(i + 1)
            self.history['log_likelihoods'].append(log_likelihood_new)
            self.history['parameter_values'].append(self.params.copy())
            
            # Check convergence
            if np.abs(log_likelihood_new - log_likelihood_old) < tol:
                return {
                    'converged': True,
                    'iterations': i + 1,
                    'final_params': self.params,
                    'final_log_likelihood': log_likelihood_new,
                    'history': self.history
                }
            
            log_likelihood_old = log_likelihood_new
        
        return {
            'converged': False,
            'iterations': max_iter,
            'final_params': self.params,
            'final_log_likelihood': log_likelihood_old,
            'history': self.history
        }

class CFR(BaseEM):
    """Case Fatality Rate estimation using EM algorithm."""
    
    def __init__(self, n: int, d: int, u: int, alpha: float = 0.5):
        """
        Initialize CFR estimation.
        
        Args:
            n: Total number of reported cases
            d: Number of reported deaths
            u: Number of cases with unknown outcomes
            alpha: Weight for reported deaths vs expected deaths
        """
        data = {'n': n, 'd': d, 'u': u}
        params = {'cfr': d/n, 'alpha': alpha}
        super().__init__(data, params)

    def e_step(self) -> float:
        """Compute expected deaths from unknown outcomes."""
        return self.data['u'] * self.params['cfr']

    def m_step(self, expected_deaths_from_unknown: float) -> Dict[str, float]:
        """Update CFR estimate."""
        alpha = self.params['alpha']
        new_cfr = (alpha * self.data['d'] + 
                  (1 - alpha) * expected_deaths_from_unknown) / self.data['n']
        return {'cfr': new_cfr}

    def log_likelihood(self) -> float:
        """Compute log-likelihood of current CFR estimate."""
        cfr = self.params['cfr']
        if cfr == 0 or cfr == 1:
            return -np.inf
        return (self.data['d'] * np.log(cfr) + 
                (self.data['n'] - self.data['d']) * np.log(1 - cfr))

class R0(BaseEM):
    """Basic Reproduction Number estimation using EM algorithm."""
    
    def __init__(self, I: np.ndarray, gamma: float, initial_R0: float):
        """
        Initialize R0 estimation.
        
        Args:
            I: Array of number of infected individuals over time
            gamma: Recovery rate
            initial_R0: Initial estimate of R0
        """
        data = {'I': np.array(I)}
        params = {'R0': initial_R0, 'gamma': gamma}
        super().__init__(data, params)

    def e_step(self) -> np.ndarray:
        """Compute expected secondary infections."""
        return (self.params['R0'] * self.data['I'] * 
                (1 - self.params['gamma']))

    def m_step(self, expected_secondary_infections: np.ndarray) -> Dict[str, float]:
        """Update R0 estimate."""
        new_R0 = np.sum(expected_secondary_infections) / np.sum(self.data['I'])
        return {'R0': new_R0}

    def log_likelihood(self) -> float:
        """Compute log-likelihood of current R0 estimate."""
        # Implement appropriate log-likelihood calculation for R0
        # This is a placeholder - you should implement the actual calculation
        return 0.0

# Example usage:
if __name__ == "__main__":
    # CFR estimation example
    cfr_model = CFR(n=1000, d=70, u=120, alpha=0.5)
    cfr_results = cfr_model.fit()
    print(f"CFR Results: {cfr_results}")

    # R0 estimation example
    infections = [10, 20, 25, 30, 40, 45]
    r0_model = R0(I=infections, gamma=0.1, initial_R0=2.0)
    r0_results = r0_model.fit()
    print(f"R0 Results: {r0_results}")