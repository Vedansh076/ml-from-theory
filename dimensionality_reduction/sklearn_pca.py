from sklearn.decomposition import PCA
#sklearn performs SVD
def sklearn_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca