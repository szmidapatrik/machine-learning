import numpy as np
from statkit.regression import LinearRegression, PolynomialRegression

class BasisFuncRegression:
    
    # Param
    basis = []
    
    # Constructor
    def __init__(self, α=1, ß=1):
        self.α = α
        self.ß = ß
        
    
    
    
    # ----- Data generation and transformation -----

    # Polynomial function
    def polynomial_function(self, x, pol_params=None):
        return PolynomialRegression(self.α, self.ß).polynomial_function(x, pol_params)
        
    # Generate data 
    def sample(self, N, min, max, μ, Σ, pol_params=None):
        '''Sample data fitting a random generated polynomial function.\n
           Params:\n
           -N: number of samples,\n
           -[min, max]: limits of the interval from which the data is sampled,\n
           -μ: mean of the noise (normal distribution),\n
           -Σ: covariance matrix of the noise (normal distribution).
        '''
        x = np.linspace(min, max, N)
        y = self.polynomial_function(x, pol_params) + np.random.normal(μ, Σ, N)
        return x, y
    
    # Transform data 
    def transform(self, X, bases, func = "gaussRBF", γ = 1, min = None, max = None):
        
        # Bases points
        bases_points = []
        
        # Min-Max setter
        if min == None:
            min = np.argmin(X)
        if max == None:
            max = np.argmax(X)
        
        # Bases setter
        if np.array(bases).shape == ():
            bases_points = np.linspace(min, max, bases)
        else:
            bases_points = bases
        
        # Bases function
        if func == "gaussRBF":
            return self.gauss_RBF(X, γ, bases_points)
        else:
            return self.gauss_RBF(X, γ, bases_points)
    
    # Gaus-RBF
    def gauss_RBF(self, X, γ, bases_points):
        Φ = np.exp(-γ * (np.abs(X[:, np.newaxis] - bases_points) ** 2) )
        return Φ 
     
    
    
    
    # ----- Maximum Likelyhood -----

    # ML (Maximum likelihood) fit
    def fit_ML(self, X, y):
        LinearRegression(self.α, self.ß).fit_ML(X, y)

    # ML (Maximum likelihood) estimate
    def predict_ML(self, X):
        return LinearRegression(self.α, self.ß).predict_ML(X)
    
    # ML (Maximum likelihood) fit-predict
    def fit_predict_ML(self, X, y):
        return LinearRegression(self.α, self.ß).fit_predict_ML(X, y)
    
    # ML (Maximum likelihood) transform-fit-predict
    def transform_fit_predict_ML(self, X, y, bases, func = "gaussRBF", γ = 1, min = None, max = None):
        Φ = self.transform(X, bases, func, γ, min, max)
        return self.fit_predict_ML(Φ, y)
    
    
    
    # ----- Maximum A Posteriori -----

    # MAP (Maximum A Posteriori) fit
    def fit_MAP(self, X, y):
        LinearRegression(self.α, self.ß).fit_MAP(X, y)

    # MAP (Maximum A Posteriori) estimate
    def predict_MAP(self, X):
        return LinearRegression(self.α, self.ß).predict_MAP(X)
    
    # MAP (Maximum A Posteriori) fit-predict
    def fit_predict_MAP(self, X, y):
        return LinearRegression(self.α, self.ß).fit_predict_MAP(X, y)
    
    # MAP (Maximum A Posteriori) transform-fit-predict
    def transform_fit_predict_MAP(self, degree, X, y):
        Φ = self.transform(X, degree)
        return self.fit_predict_MAP(Φ, y)
    
    