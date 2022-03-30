# library that contains matrix
# operations required for the
# lab.

import numpy as np


def det(matrix: np.ndarray):
    return np.linalg.det(matrix)


def create_random_matrix(n: int):
    # Creates a random matrix (suddenly :D)

    matrix = np.random.randint(-8, 0, (n, n))
    matrix = matrix.astype('float64')

    return matrix


def lu_decomposition(matrix: np.ndarray):
    # does the LU-decomposition and
    # returns U-matrix and L-matrix

    u_matrix = np.zeros(matrix.shape, dtype=float)
    l_matrix = np.identity(matrix.shape[0], dtype=float)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i <= j:
                u_matrix[i, j] = matrix[i, j] - np.dot(l_matrix[i, :i], u_matrix[:i, j])
            if i > j:
                l_matrix[i, j] = (matrix[i, j] - np.dot(l_matrix[i, :j], u_matrix[:j, j])) / u_matrix[j, j]

    lu = (u_matrix, l_matrix)
    return lu
