import numpy as np
import matplotlib.pyplot as plt

class LaplaceDistribution:
    
    # Params
    μ = 0
    b = 1

    # Constructor
    def __init__(self, μ, b):
        self.μ = μ
        self.b = b

    # Sample from distibution
    def sample(self, size):
        return np.random.laplace(self.μ, self.b, size)
    
    # PDF - Probability density function
    def pdf(self, axis):
        return 1 / (2 * self.b) * np.exp( -np.abs(((axis - self.μ) / self.b)))
    
    # CDF - Cumultative Density Function
    def cdf(self, axis):
        cdf_values = np.cumsum(self.pdf(axis))
        min_cdf = cdf_values.min()
        max_cdf = cdf_values.max()
        normalized_cdf = (cdf_values - min_cdf) / (max_cdf - min_cdf)
        return normalized_cdf

    # Array of zeros with size param
    def zeros(self, size):
        return np.zeros(size)