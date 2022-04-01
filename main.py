import moperations as m
import numpy as np


def first_part():
    pass


def second_part():
    # First task
    matrix = m.create_random_matrix(6)
    print(f'{matrix}\n')
    umatrix = m.lu_decomposition(matrix)[0]
    lmatrix = m.lu_decomposition(matrix)[1]
    # if A = LU then detA = detL x detU
    print(
        f'detA = {m.det(matrix)} = detL x detU = '
        f'{m.det(lmatrix)} x {m.det(umatrix)} = {m.det(lmatrix)*m.det(umatrix)}\n')

    # Second task
    q_matrix = m.qr_decomposition(matrix)[0]
    r_matrix = m.qr_decomposition(matrix)[1]
    print(f'{q_matrix}\n\n{r_matrix}\n\n{m.mul(q_matrix, r_matrix)}\n\n')
    q_matrix = m.numpy_qr_decomposition(matrix)[0]
    r_matrix = m.numpy_qr_decomposition(matrix)[1]
    print(
        f'numpy.linealg.qr:\n\n{q_matrix}\n\n{r_matrix}\n\n{m.mul(q_matrix, r_matrix)}\n')

    # Third task
    equation_system = np.array([[3.3, 2.1, 2.8, 0.8],
                                [4.1, 3.7, 4.8, 5.7],
                                [2.7, 1.8, 1.1, 3.2]])
    solution = m.simple_iteration_method(equation_system)
    print(f'solution of linear algebraic system: {solution}')


if __name__ == '__main__':
    second_part()
