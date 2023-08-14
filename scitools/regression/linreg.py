import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    
    # ----- Params -----

    m = 1
    b = 0

    α = 1
    ß = 1
    λ = 1

    wµ = 0
    wΣ = 0

    # Fit params
    w_ML = []
    w_MAP = []


    # Constructor
    def __init__(self, α=1, ß=1):
        self.α = α
        self.ß = ß
        self.λ = self.α / self.ß



    # ----- Data generation -----

    # Linear function
    def linear_function(self, x, m = None, b = None):
        if m != None:
            self.m = m
        if b != None:
            self.b = b
        return self.m * x + self.b
    
    # Generate data
    def sample(self, N, min, max, m, b, μ, Σ):
        '''Sample data fitting the given linear function.\n
           Params:\n
           -N: number of samples,\n
           -[min, max]: limits of the interval from which the data is sampled,\n
           -m: gradient of the linear function,\n
           -b: bias of the linear function,\n
           -μ: mean of the noise (normal distribution),\n
           -Σ: covariance matrix of the noise (normal distribution).
        '''
        self.m = m
        self.b = b
        x = np.linspace(min, max, N)
        y = self.linear_function(x, m, b) + np.random.normal(μ, Σ, N)
        return np.vstack([x,y]).T
    
    # transformation
    def Φ(self, X):
        return np.vstack([np.ones(len(X)), X.T]).T

    # ----- Maximum Likelihood -----

    # ML (Maximum likelihood) fit
    def fit_ML(self, X, y):
        Φ = self.Φ(X)
        self.w_ML = np.linalg.solve(Φ.T @ Φ, Φ.T @ y)

    # ML (Maximum likelihood) estimate
    def predict_ML(self, X):
        Φ = self.Φ(X)
        return Φ @ self.w_ML
    
    # ML (Maximum likelihood) fit-predict
    def fit_predict_ML(self, X, y):
        self.fit_ML(X, y)
        return self.predict_ML(X)



    # ----- Maximum A Posteriori -----
    
    # MAP (Maximum A Posteriori) fit
    def fit_MAP(self, X, y):
        Φ = self.Φ(X)
        I = np.identity(2)
        self.w_MAP = np.linalg.solve(Φ.T @ Φ + self.λ * I, Φ.T @ y)

    # MAP (Maximum A Posteriori) estimate
    def predict_MAP(self, X):
        Φ = self.Φ(X)
        return Φ @ self.w_MAP
    
    # MAP (Maximum A Posteriori) fit-predict
    def fit_predict_MAP(self, X, y):
        self.fit_MAP(X, y)
        return self.predict_MAP(X)
    


    # ----- Bayesian Linear Regression
    
    # w posterior
    def w_posterior(self, X, y):
        Φ = self.Φ(X)
        I = np.identity(2)
        self.wΣ = self.ß * Φ.T @ Φ + self.α * I
        self.wμ = pow(self.wΣ, -1) @ (self.ß * Φ.T @ y)

    # Bayesian predict
    def bayesian_predict(self, X, y):
        Φ = np.vstack([np.ones(len(X)), X])
        self.w_posterior(X,y)
        return self.wΣ @ Φ, self.wμ @ Φ