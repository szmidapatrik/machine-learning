import numpy as np
import matplotlib.pyplot as plt

class NormalDistribution:
    
    # Params
    μ = 0
    σ = 1

    # Constructor
    def __init__(self, μ, σ):
        self.μ = μ
        self.σ = σ

    # Sample from distibution
    def sample(self, size):
        return np.random.normal(self.μ, self.σ ** 2, size)
    
    # PDF - Probability density function
    def pdf(self, axis):
        return (1 / self.σ * np.sqrt(2 * np.pi)) * np.exp( -(1/2) * (((axis - self.μ) / self.σ) ** 2) )
    
    # CDF - Cumultative Density Function
    def cdf(self, axis):
        return np.cumsum(self.pdf(axis))

    # Array of zeros with size param
    def zeros(self, size):
        return np.zeros(size)