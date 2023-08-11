import numpy as np
import matplotlib.pyplot as plt

class ExponentialDistribution:
    
    # Params
    λ = []

    # Constructor
    def __init__(self, λ):
        if λ < 0:
            raise Exception("Invalid parameter: λ must be non-negative.")
        self.λ = λ

    # Sample from distibution
    def sample(self, size):
        return np.random.exponential(self.λ, size)
    
    # Exponential function
    def pdf_value(self, value):
        if value < 0:
            return 0
        else:
            return self.λ * np.exp(-self.λ * value)

    # PDF - Probability density function
    def pdf(self, axis):
        return np.concatenate([np.zeros(len(axis[axis<0])), self.λ * np.exp(-self.λ * axis[axis >= 0])])
    
    # CDF - Cumultative Density Function
    def cdf(self, axis):
        return np.cumsum(self.pdf(axis))

    # Array of zeros with size param
    def zeros(self, size):
        return np.zeros(size)