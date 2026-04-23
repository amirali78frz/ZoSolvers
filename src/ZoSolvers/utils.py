import numpy as np


def is_diagonal(matrix, tol=1e-12):
    if matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError("Input should be a square matrix")
    return np.allclose(matrix, np.diag(np.diag(matrix)), atol=tol)
