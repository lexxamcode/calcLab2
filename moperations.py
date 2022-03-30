# library that contains matrix
# operations required for the
# lab.

import numpy as np


def det(matrix: np.ndarray):
    return np.linalg.det(matrix)


def mul(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b)


def proj(u: np.ndarray, a: np.ndarray):
    return np.dot(((np.dot(u, a)) / (np.dot(u, u))), u)


def norm(vector: np.ndarray):
    return np.linalg.norm(vector)


def numpy_qr_decomposition(matrix: np.ndarray):
    return np.linalg.qr(matrix)


def qr_decomposition(matrix: np.ndarray):
    # Gram-Schmidt process.
    # Firstly, we need to divide columns
    # of matrix
    a1 = matrix[:, 0]
    a2 = matrix[:, 1]
    a3 = matrix[:, 2]
    a4 = matrix[:, 3]
    a5 = matrix[:, 4]
    a6 = matrix[:, 5]

    u1 = a1
    u2 = a2 - proj(u1, a2)
    u3 = a3 - proj(u1, a3) - proj(u2, a3)
    u4 = a4 - proj(u1, a4) - proj(u2, a4) - proj(u3, a4)
    u5 = a5 - proj(u1, a5) - proj(u2, a5) - proj(u3, a5) - proj(u4, a5)
    u6 = a6 - proj(u1, a6) - proj(u2, a6) - proj(u3, a6) - \
        proj(u4, a6) - proj(u5, a6)

    e1 = u1 / norm(u1)
    e2 = u2 / norm(u2)
    e3 = u3 / norm(u3)
    e4 = u4 / norm(u4)
    e5 = u5 / norm(u5)
    e6 = u6 / norm(u6)

    # now express matrix with new orthonormal basis
    a1 = np.dot(np.dot(e1, a1), e1)
    a2 = np.dot(np.dot(e1, a2), e1) + np.dot(np.dot(e2, a2), e2)
    a3 = np.dot(np.dot(e1, a3), e1) + np.dot(np.dot(e2, a3), e2) + \
        np.dot(np.dot(e3, a3), e3)
    a4 = np.dot(np.dot(e1, a4), e1) + np.dot(np.dot(e2, a4), e2) + \
        np.dot(np.dot(e3, a4), e3) + np.dot(np.dot(e4, a4), e4)
    a5 = np.dot(np.dot(e1, a5), e1) + np.dot(np.dot(e2, a5), e2) + np.dot(np.dot(e3, a5), e3) + \
        np.dot(np.dot(e4, a5), e4) + np.dot(np.dot(e5, a5), e5)

    # A = QR
    # Q = [e1, e2, e3, e4, e5]

    q_matrix = np.array([e1, e2, e3, e4, e5, e6])
    q_matrix = q_matrix.transpose()

    r_matrix = np.zeros(matrix.shape, dtype=float)
    r_matrix[0, 0] = np.dot(e1, a1)
    r_matrix[0, 1] = np.dot(e1, a2)
    r_matrix[0, 2] = np.dot(e1, a3)
    r_matrix[0, 3] = np.dot(e1, a4)
    r_matrix[0, 4] = np.dot(e1, a5)
    r_matrix[0, 5] = np.dot(e1, a6)

    r_matrix[1, 1] = np.dot(e2, a2)
    r_matrix[1, 2] = np.dot(e2, a3)
    r_matrix[1, 3] = np.dot(e2, a4)
    r_matrix[1, 4] = np.dot(e2, a5)
    r_matrix[1, 5] = np.dot(e2, a6)

    r_matrix[2, 2] = np.dot(e3, a3)
    r_matrix[2, 3] = np.dot(e3, a4)
    r_matrix[2, 4] = np.dot(e3, a5)
    r_matrix[2, 5] = np.dot(e3, a6)

    r_matrix[3, 3] = np.dot(e4, a4)
    r_matrix[3, 4] = np.dot(e4, a5)
    r_matrix[3, 5] = np.dot(e4, a6)

    r_matrix[4, 4] = np.dot(e5, a5)
    r_matrix[4, 5] = np.dot(e5, a6)

    r_matrix[5, 5] = np.dot(e6, a6)

    qr = (q_matrix, r_matrix)
    return qr


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
                u_matrix[i, j] = matrix[i, j] - \
                    np.dot(l_matrix[i, :i], u_matrix[:i, j])
            if i > j:
                l_matrix[i, j] = (
                    matrix[i, j] - np.dot(l_matrix[i, :j], u_matrix[:j, j])) / u_matrix[j, j]

    lu = (u_matrix, l_matrix)
    return lu
