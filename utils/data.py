import numpy as np

class CFR():
    def __init__(self, n, d, u, alpha):
        self.n = n  # total number of reported cases
        self.d = d  # number of reported deaths
        self.u = u  # number of cases with unknown outcomes
        self.cfr = d/n  # intial estimate of CFR
        self.alpha = alpha

    def e_step(self):
        expected_deaths_from_unknown = self.u * self.cfr
        return expected_deaths_from_unknown

    def m_step(self, expected_deaths_from_unknown, alpha):
        new_cfr = (alpha*self.d + (1-alpha)*expected_deaths_from_unknown)/self.n
        return new_cfr

    def log_likelihood(self):
        if self.cfr == 0 or self.cfr == 1:
            return -np.inf
        log_likelihood = self.d * np.log(self.cfr) + (self.n - self.d) * np.log(1 - self.cfr)
        return log_likelihood

    def fit(self, tol=1e-6, max_iter=100):
        log_likelihood_old = self.log_likelihood()
        for i in range(max_iter):
            expected_deaths_from_unknown = self.e_step()
            new_cfr = self.m_step(expected_deaths_from_unknown)
            self.cfr = new_cfr
            log_likelihood_new = self.log_likelihood()
            if np.abs(log_likelihood_new - log_likelihood_old) < tol:
                break
            log_likelihood_old = log_likelihood_new
        return self.cfr

# # Example usage:
# # Define initial parameters
# n = 1000  # Total number of reported cases
# d = 70    # Number of reported deaths
# u = 120   # Number of cases with unknown outcomes

# # Instantiate the EM algorithm
# em = CFR(n, d, u)

# # Fit the model to the data
# final_cfr = em.fit()
# print(f"Estimated CFR: {final_cfr:.4f}")

class R0():
    def __init__(self, I, gamma, R0):
        self.I  = np.array(I) # Number of infected individuals over time
        self.R0 = R0
        self.gamma = gamma  # recovery rate

    def e_step(self):
        expected_secondary_infections = self.R0 * self.I * (1-self.gamma)
        return expected_secondary_infections

    def m_step(self, expected_secondary_infections):
        new_R0 = np.sum(expected_secondary_infections)/np.sum(self.I)
        return new_R0

    def fit(self, tol=1e-6, max_iter=100):
        for i in range(max_iter):
            expected_secondary_infections = self.e_step()
            new_R0 = self.m_step(expected_secondary_infections)
            if np.abs(new_R0 - self.R0) < tol:
                break
            self.R0 = new_R0
        return self.R0

# # Infection over time
# infections = [10, 20, 25, 30, 40, 45]
# gamma = 0.1  # Example recovery rate
# initial_R0 = 2.0  # Initial estimate of R0

# # Instantiate the EM algorithm
# em_r0 = R0(I=infections, gamma=gamma, R0 = initial_R0)

# # Fit the model to the data
# final_R0 = em_r0.fit()
# print(f"Estimated R0: {final_R0:.4f}")