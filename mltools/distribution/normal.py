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

    # Sample from distibution using Box-Müller Transformation
    def sample(self, size):
        N = size if size % 2 == 0 else size + 1
        U = np.random.rand(N)
        U_1 = U[:int(N/2)]
        U_2 = U[int(N/2):]
        X_1 = np.sqrt(-2 * np.log(U_1)) * np.cos(2 * np.pi * U_2)
        X_2 = np.sqrt(-2 * np.log(U_1)) * np.sin(2 * np.pi * U_2)
        X = np.concatenate([X_1, X_2])
        return X if size % 1 == 0 else X[:-1]
    
    # PDF - Probability density function
    def pdf(self, axis):
        return (1 / (self.σ * np.sqrt(2 * np.pi))) * np.exp( -(1/2) * (((axis - self.μ) / self.σ) ** 2))
    
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