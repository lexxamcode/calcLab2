import moperations as m


def main():
    # First task
    matrix = m.create_random_matrix(6)
    print(f'{matrix}\n')
    umatrix = m.lu_decomposition(matrix)[0]
    lmatrix = m.lu_decomposition(matrix)[1]
    # if A = LU then detA = detL x detU
    print(f'detA = {m.det(matrix)} = detL x detU = '
          f'{m.det(lmatrix)} x {m.det(umatrix)} = {m.det(lmatrix)*m.det(umatrix)}\n')

    # Second task
    q_matrix = m.qr_decomposition(matrix)[0]
    r_matrix = m.qr_decomposition(matrix)[1]
    print(f'{q_matrix}\n\n{r_matrix}\n\n{m.mul(q_matrix, r_matrix)}\n\n')
    q_matrix = m.numpy_qr_decomposition(matrix)[0]
    r_matrix = m.numpy_qr_decomposition(matrix)[1]
    print(f'numpy.linealg.qr:\n\n{q_matrix}\n\n{r_matrix}\n\n{m.mul(q_matrix, r_matrix)}\n')


if __name__ == '__main__':
    main()
