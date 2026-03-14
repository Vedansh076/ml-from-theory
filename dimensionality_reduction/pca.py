import numpy as np
class MyPCA:
    def __init__(self, n_components):
        # number of principal components we want to keep
        self.n_components = n_components

        # will store the principal directions
        self.components = None

        # will store the mean of each feature
        self.mean = None
    
    def fit(self,X):
        self.mean = np.mean(X,axis= 0)
        X_centered = X - self.mean
        #Covariance tells us how features vary together.
        cov_matrix = np.cov(X_centered, rowvar=False) 

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        '''eigenvectors → directions of data spread
           eigenvalues → amount of variance in those directions'''
        
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        self.components = eigenvectors[:, :self.n_components] #select top n max variance directions
    
    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components #project onto the max-variance directions
    
    #for convenience combining both fit and transform
    def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)