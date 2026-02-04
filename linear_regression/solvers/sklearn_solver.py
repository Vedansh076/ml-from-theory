from sklearn.linear_model import LinearRegression

def solve_sklearn(X, y):
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    return model.coef_, model.predict(X)
