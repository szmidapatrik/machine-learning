import numpy as np
from scitools.regression import LinearRegression

class PolynomialRegression:
    
    # Constructor
    def __init__(self, α=1, ß=1):
        self.α = α
        self.ß = ß

        
    # ----- Data generation and transformation -----

    # Polynomial function
    def polynomial_function(self, x, pol_params=None):
        if pol_params != None:
            if np.array(pol_params).shape == (3,):
                params = pol_params
            elif np.array(pol_params).shape == ():
                params = np.ones(3) * pol_params
            else:
                print('The given @pol_params parameter\'s type was incorrect. It must either be an integer, float, or an array of numbers with the length of 3.\nA random polynom will be generated.')
                params = np.random.randint(-10,10,3)
        else:
            params = np.random.randint(-10,10,3)
        return (x + params[0]) * (x + params[1]) * (x + params[2])
        
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
        return np.vstack([x,y]).T
    
    # Transform data 
    def transform(self, X, degree):
        if degree < 1:
            raise Exception('The @degree parameter must be equal to or greater than 1.')
        if degree % 1 != 0:
            raise Exception('The @degree parameter must be an integer.')
        if degree == 1:
            print('The @degree parameter with value 1 applies no transformation to the data.')
            return X
        Φ = X
        for d in range(2, degree + 1):
            Φ = np.vstack([Φ, X ** d])
        return Φ.T
    
    
    
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
    
    