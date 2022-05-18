import moperations as m
import numpy as np


def first_part():
    print('\n\nFirst Part\n\n')
    # First task
    print('First Task\n')
    matrix = np.random.randint(-3, 3, (10, 10))
    print(f'Random 10x10 matrix:\n{matrix}')
    minor = matrix[1:5, 6:10]
    print(f'Minor of the 4th order:\n{minor}')
    print(f'Det of the minor is {np.linalg.det(minor)}')

    # Second task
    print('Second Task\n')
    f_matrix = np.random.sample((4, 4)) + 2
    s_matrix = np.random.sample((4, 4)) + 2
    print(f'First matrix:\n{f_matrix}')
    print(f'Second matrix:\n{s_matrix}')
    print(f'Vector multiply:\n{m.vector_mul(f_matrix, s_matrix)}')
    print(f'Matrix multiply:\n{m.matrix_algorithm(f_matrix, s_matrix)}')
    print(f'numpy.dot:\n{np.dot(f_matrix, s_matrix)}')

    # Third task
    print('Third Task\n')
    triangle_matrix = np.random.randint(-101, 101, (5, 5))
    triangle_matrix = triangle_matrix.astype('float64')
    triangle_matrix = np.triu(triangle_matrix)
    print('Upper triangular matrix system solution:\n')
    print('A:')
    print(triangle_matrix)

    b_vector = np.random.randint(-101, 101, (5, 1))
    b_vector = b_vector.astype('float64')
    print('B:')
    print(b_vector)
    print(f'Solution:\n{m.tri_solve(triangle_matrix, b_vector)}')

    # Fourth task
    print('Fourth Task\n')
    matrix = np.array([[4.4, -2.5, 19.2, -10.8],
                       [5.5, -9.3, -14.2, 13.2],
                       [7.1, -11.5, 5.3, -6.7],
                       [14.2, 23.4, -8.8, 5.3]])
    b_vector = np.array([4.3, 6.8, -1.8, 7.2])
    result = m.lu_system_solve(matrix, b_vector)
    print(f'Matrix:\n{matrix}\nB vector:\n{b_vector}\nX vector:\n{result}')


def second_part():
    print('\nSecond Part\n')
    # First task
    print('First task')
    matrix = m.create_random_matrix(6)
    print(f'{matrix}\n')
    lmatrix = m.lu_decomposition(matrix)[0]
    umatrix = m.lu_decomposition(matrix)[1]

    # if A = LU then detA = detL x detU
    print(
        f'detA = {m.det(matrix)} = detL x detU = '
        f'{m.det(lmatrix)} x {m.det(umatrix)} = {m.det(lmatrix)*m.det(umatrix)}\n')

    # Second task
    print('Second task')
    q_matrix = m.gram_schmidt_qr(matrix)[0]
    r_matrix = m.gram_schmidt_qr(matrix)[1]
    print(f'{q_matrix}\n\n{r_matrix}\n\n{m.mul(q_matrix, r_matrix)}\n\n')
    q_matrix = m.numpy_qr_decomposition(matrix)[0]
    r_matrix = m.numpy_qr_decomposition(matrix)[1]
    print(
        f'numpy.linealg.qr:\n\n{q_matrix}\n\n{r_matrix}\n\n{m.mul(q_matrix, r_matrix)}\n')

    # Third task
    print('Third task')
    equation_system = np.array([[3.3, 2.1, 0.8, 0.8],
                                [4.1, 11.2, 4.8, 5.7],
                                [2.7, 1.8, 5.1, 3.2]])
    solution = m.simple_iteration_method(equation_system)
    print(f'solution of linear algebraic system: {solution}\n')


def third_part():
    matrix = np.random.randint(-11, 11, (6, 6))
    # print(f'\nThird Part\n{matrix}\n--Q--\n')
    #print(f'{m.householder_qr(matrix)[0]}\n')
    # print(f'{m.gram_schmidt_qr(matrix)[0]}\n')
    # print(f'{m.givens_qr(matrix)[0]}\n')
    # print(f'{np.linalg.qr(matrix)[0]}\n')

    # print(f'--R--\n{m.householder_qr(matrix)[1]}\n')
    # print(f'{m.gram_schmidt_qr(matrix)[1]}\n')
    # print(f'{m.givens_qr(matrix)[1]}\n')
    # print(f'{np.linalg.qr(matrix)[1]}\n')
    pass

def main():
    first_part()
    second_part()
    third_part()


if __name__ == '__main__':
    main()
