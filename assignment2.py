def affine(x, W, b):
    """Apply affine transformation to `x` by computing `Wx + b` and return
    a list with the dimensions of `b`.

    Args:
        x (list): List that is to be transformed. 1 dimensional.
        W (list): List representing the transformation. 2 dimensional.
        b (list): List that is added to the transformation. 1 dimensional.

    Returns:
        list: Affine transformation list with the same dimensions as `b`.
    """
    assert len(b) == len(W), \
        'Dimension of b must match the number of rows in W: {} != {}'.format(len(b), len(W))
    assert len(x) == len(W[0]), \
        'Dimension of x must match the number of columns in W: {} != {}'.format(len(x), len(W[0]))

    y = []

    for i, W_i in enumerate(W):
        y_i = sum(w_ij * x_i for w_ij, x_i in zip(W_i, x))
        y_i += b[i]
        y.append(y_i)

    return y


if __name__ == '__main__':
    # Test the `affine` function.
    x = [7, 12, 4, -5]
    W = [[3, 2, -8, 1], [3, 8, -1, 4], [-4, 5, 9, -2]]
    b = [0.1, 0.3, 0.5]
    t = [8.1, 93.3, 78.5]

    y = affine(x, W, b)

    assert len(b) == len(y)

    for y_i, t_i in zip(y, t):
        assert y_i == t_i
