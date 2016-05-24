def outer(x, y):
    """Compute the outer product of two one-dimensional lists and return a
    two-dimensional list with the shape (length of `x`, length of `y`).

    Args:
        x (list): First list, treated as a column vector. 1 dimensional.
        y (list): Second list, treated as a row vector. 1 dimensional.

    Returns:
        list: Outer product between x and y.
    """
    return [[x_i * y_i for y_i in y] for x_i in x]


if __name__ == '__main__':
    # Test the `outer` function.
    x = [2, 7, 1, 5]
    y = [6, 3, 9]

    A = outer(x, y)

    assert len(x) == len(A)
    assert len(y) == len(A[0])

    for i, A_i in enumerate(A):
        for j, a_ij in enumerate(A_i):
            assert x[i] * y[j] == a_ij
