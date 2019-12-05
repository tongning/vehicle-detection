import numpy as np
from scipy.stats import multivariate_normal

x, y = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j]

# Need an (N, 2) array of (x, y) pairs.
xy = np.column_stack([x.flat, y.flat])

mu = np.array([0.0, 0.0])

sigma = np.array([20, 20])
covariance = np.diag(sigma**2)

z = multivariate_normal.pdf([[3,2]], mean=mu, cov=covariance)
print(z)