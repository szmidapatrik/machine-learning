from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import collections
import numpy as np
import random

class NaivBayes:
    
    # Params
    μ = []
    σ = []

    K = 0
    prior_y = []
        
    # Generate data with K number of classes (only 3 works)
    def sample(self, K, class_N):
        s_1 = random.randint(round(class_N * 0.9), round(class_N * 1.1))
        s_2 = random.randint(round(class_N * 0.9), round(class_N * 1.1))
        s_3 = random.randint(round(class_N * 0.9), round(class_N * 1.1))
        
        x_1 = np.random.multivariate_normal([2,2], [[1,0],[0,1]], s_1)
        x_2 = np.random.multivariate_normal([4,5], [[1,0],[0,1]], s_2)
        x_3 = np.random.multivariate_normal([6,3], [[1,0],[0,1]], s_3)
        
        # X and y
        X = np.concatenate([x_1, x_2, x_3])
        y = np.concatenate([np.zeros(s_1), np.ones(s_2), 2 * np.ones(s_3)])
        
        return X, y

    # MAP (Maximum A Posteriori) - Learn the parameters of the data
    def fit(self, X, y):
        
        # Set self params
        self.K = np.unique(y).size
        self.prior_y = collections.Counter(y)

        # Train the model
        for k in range(self.K):
            self.μ.append(np.mean(X[y == k], axis=0))
            self.σ.append(np.var(X[y == k]))
    
    # MAP (Maximum A Posteriori) - Predict the class of the given data points
    def predict(self, X):
        posterior = np.zeros([len(X), 3])
        for k in range(self.K):
            posterior[:,k] = multivariate_normal(self.μ[k], self.σ[2]).pdf(X) * self.prior_y[k]
        posterior = np.argmax(posterior,axis=1)
        return posterior
    
    # MAP (Maximum A Posteriori) - Fit predict
    def fit_predict(self, X, y):
        self.fit(X,y)
        return self.predict(X)