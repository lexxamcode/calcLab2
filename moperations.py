# library that contains matrix
# operations required for the
# lab.
import numpy as np


def tri_solve(triu_matrix: np.ndarray, b: np.ndarray):
    # AX = B, where A - upper triangle matrix
    x = np.zeros(triu_matrix.shape[0], dtype=float)
    x[triu_matrix.shape[0] - 1] = b[triu_matrix.shape[0] - 1]

    for i in range(triu_matrix.shape[0]):
        b[i] = b[i] / triu_matrix[i, i]
        triu_matrix[i, :] = triu_matrix[i, :]/triu_matrix[i, i]

    for i in range(triu_matrix.shape[0] - 1, -1, -1):
        x[i] = b[i]

        for j in range(i+1, triu_matrix.shape[1]):
            koef = triu_matrix[i, j]
            x[i] -= koef*x[j]
    return x


def matrix_algorithm(a: np.ndarray, b: np.ndarray):
    result = np.zeros((a.shape[0], b.shape[1]), dtype=float)
    if a.shape[1] != b.shape[0]:
        print('arguments have a different shape')
        return
    else:
        for i in range(a.shape[0]):
            result[:, i] = mul(a, b[:, i])
    return result


def vector_mul(a: np.ndarray, b: np.ndarray):
    result = np.zeros((a.shape[0], b.shape[1]), dtype=float)
    if a.shape[1] != b.shape[0]:
        print('arguments have a different shape')
        return
    else:
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                result[i, j] = np.dot(a[i, :], b[:, j])

    return result


def find_abs_max(vector: np.array):
    max_element = vector[0]
    for i in range(vector.shape[0]):
        if abs(vector[i]) > max_element:
            max_element = vector[i]

    return abs(max_element)


def simple_iteration_method(matrix: np.ndarray):
    # Check
    correct = True
    for i in range(matrix.shape[0]):
        sum_others = 0
        for j in range(matrix.shape[0]):
            if i != j:
                sum_others = sum_others + abs(matrix[i, j])
        if abs(matrix[i, i]) <= sum_others:
            correct = False

    if correct:
        print('\nCorrect system\n')
    else:
        print('\nIncorrect system\n')
        return

    # Jacobi method: X = CX + F
    x_vector = np.zeros(matrix.shape[0])
    f_vector = np.zeros(matrix.shape[0])
    # e_matrix = np.eye(matrix.shape[0] - 1)
    eps = 0.001

    # Creating C matrix:
    c_matrix = np.zeros((matrix.shape[0], matrix.shape[0]), dtype='float64')
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[0]):
            if i != j:
                c_matrix[i, j] = -(matrix[i, j]/matrix[i, i])

    for i in range(c_matrix.shape[0]):
        f_vector[i] = matrix[i, matrix.shape[1] - 1]/matrix[i, i]
    # print(f_vector)
    # print(c_matrix)
    # print(x_vector)

    while True:
        temp_vector = x_vector
        x_vector = mul(c_matrix, x_vector.transpose()) + f_vector
        # print(x_vector)
        if abs(find_abs_max(x_vector) - find_abs_max(temp_vector)) < eps:
            break
    # print(x_vector)
    return x_vector


def det(matrix: np.ndarray):
    # Matrix' determinant
    return np.linalg.det(matrix)


def mul(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b)


def proj(u: np.ndarray, a: np.ndarray):
    return np.dot(((np.dot(u, a)) / (np.dot(u, u))), u)


def norm(vector: np.ndarray):
    # Norm of the vector
    return np.linalg.norm(vector)


def numpy_qr_decomposition(matrix: np.ndarray):
    return np.linalg.qr(matrix)


def gram_schmidt_qr(matrix: np.ndarray):
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
    a5 = np.dot(np.dot(e1, a5), e1) + np.dot(np.dot(e2, a5), e2) + np.dot(np.dot(e3, a5), e3) +\
        np.dot(np.dot(e4, a5), e4) + np.dot(np.dot(e5, a5), e5)
    a6 = np.dot(np.dot(e1, a6), e1) + np.dot(np.dot(e2, a6), e2) + np.dot(np.dot(e3, a6), e3) + \
        np.dot(np.dot(e4, a6), e4) + np.dot(np.dot(e5, a6), e5) + np.dot(np.dot(e6, a6), e6)
    # A = QR
    # Q = [e1, e2, e3, e4, e5, e6]

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

    lu = (l_matrix, u_matrix)
    return lu


def tri_solve_l(matrix: np.ndarray, b: np.ndarray):
    x = np.zeros((matrix.shape[0], 1), float)
    x[0] = b[0]

    for i in range(1, matrix.shape[0]):
        x[i] = b[i] - np.dot(matrix[i, :i], x[:i])
    return x


def lu_system_solve(matrix: np.ndarray, b: np.ndarray):
    if (matrix[0, 0] == 0) or (np.linalg.det(matrix[0:1, 0:1]) == 0) or (np.linalg.det(matrix[0:2, 0:2]) == 0) or \
       (np.linalg.det(matrix[0:3, 0:3]) == 0):
        print('Can\'t solve the system')
        return
    else:
        # AX = B
        # LUX = B
        # Ly = B; y = L^-1 B
        # y = UX; X = U^-1 y
        l_matrix = lu_decomposition(matrix)[0]
        u_matrix = lu_decomposition(matrix)[1]
        y = tri_solve_l(l_matrix, b)
        x = tri_solve(u_matrix, y)
        print(f'LU-decomposition:\nL:\n{l_matrix}\nU:\n{u_matrix}\ny:\n{y}')
        return x


def householder_qr(matrix: np.array):
    q_matrix = np.identity(matrix.shape[0])
    r_matrix = np.copy(matrix)
    for i in range(matrix.shape[0] - 1):
        x = r_matrix[i:, i]
        e = np.zeros_like(x)
        e[0] = norm(x)
        u = x - e
        v = u/norm(u)
        q_i = np.identity(matrix.shape[0])
        q_i[i:, i:] -= 2*np.outer(v, v)
        r_matrix = np.dot(q_i, r_matrix)
        q_matrix = np.dot(q_matrix, q_i)
    pair = (q_matrix, r_matrix)
    return pair


def givens_qr(matrix: np.array):
    n, m = matrix.shape
    q_matrix = np.identity(n)
    r_matrix = np.copy(matrix)

    (rows, cols) = np.tril_indices(n, -1, m)
    for (row, col) in zip(rows, cols):
        if r_matrix[row, col] != 0:
            r = np.hypot(r_matrix[col, col], r_matrix[row, col])
            c = r_matrix[col, col]/r
            s = -r_matrix[row, col]/r
            g = np.identity(n)
            g[[col, row], [col, row]] = c
            g[row, col] = s
            g[col, row] = -s
            r_matrix = np.dot(g, r_matrix)
            q_matrix = np.dot(q_matrix, g.T)
    qr = (q_matrix, r_matrix)
    return qr
