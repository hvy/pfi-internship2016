from assignment1 import outer
from assignment2 import affine


def forward(x, W1, b1, W2, b2):
    """Forward propagation of a neural network. Return the affine
    transformation of `x` through one hidden layer and an ouput layer.

    Args:
        x (list): Input vector. 1 dimensional.
        W1 (list): First weight matrix 2 dimensional.
        b1 (list): First bias vector. 1 dimensional.
        W2 (list): Second weight matrix. 2 dimensional.
        b2 (list): Second bias matrix. 1 dimensional.

    Returns:
        list: Weighted output. 1 dimensional.
    """
    h = affine(x, W1, b1)
    y = affine(h, W2, b2)

    return h, y


def backward(x, h, y, W1, b1, W2, b2):
    """Backward propagation of a neural network. Return the gradients of the
    parameters of the weights and biases.

    Args:
        x (list): Input vector to the first layer. 1 dimensional.
        h (list): Output of the hidden layer. 1 dimensional.
        y (list): Ouput of the network. 1 dimensional.
        W1 (list): First weight matrix 2 dimensional.
        b1 (list): First bias vector. 1 dimensional.
        W2 (list): Second weight matrix. 2 dimensional.
        b2 (list): Second bias matrix. 1 dimensional.

    Returns:
        list: Gradient of `W1`.
        list: Gradient of `b1`.
        list: Gradient of `W2`.
        list: Gradient of `b2`.
    """
    # gy = y - x
    gy = [y_i - x_i for y_i, x_i in zip(y, x)]

    gW2 = outer(gy, h)
    gb2 = gy

    # gh = W2T * gy
    gh = []
    for W2T_i in transpose(W2):
        gh_i = sum(w_ij * gy_i for w_ij, gy_i in zip(W2T_i, gy))
        gh.append(gh_i)

    gW1 = outer(gh, x)
    gb1 = gh

    return gW1, gb1, gW2, gb2


def mean_squared_error(y, t, scale=0.5):
    """Return the mean squared error (MSE). The loss is scaled with a factor,
    with a default value of 0.5.

    Args:
        y (list): Actual outputs. 1 dimensional.
        t (list): Target outputs. Same dimensions as `y`.

    Returns:
        list: Scaled MSE of the two given lists.
    """
    assert len(y) == len(t)

    return scale * sum((y_i - t_i) ** 2 for y_i, t_i in zip(y, t))


def optimize_sgd(params, grads, eta):
    """Return the optimized parameters with respect to corresponding gradients
    and a learning rate scalar.

    Args:
        params (list): Parameters to optimize. 1 or 2 dimensional.
        grads (list): Gradients to apply to the params. Same dimensions as
            `params`.
        eta (float): Learning rate.

    Returns:
       list: Optimized params. Same dimensions as `params`.
    """
    return sub(params, scalar_mul(eta, grads))


def scalar_mul(eta, x):
    """Scalar multiplication. Return a list with the same dimensions as `x`
    where each element in `x` is multiplied by `eta`.

    Args:
        x (list): Vector to scale. 1 or 2 dimensional.
        eta (float) Scalar to apply to `x`.

    Returns:
        list: A scaled list. Same dimensions as `x`.

    """
    if isinstance(x[0], list):  # 2 dimensional
        return [[eta * x_ij for x_ij in x_i] for x_i in x]
    else:  # 1 dimensional
        return [eta * x_i for x_i in x]


def sub(x1, x2):
    """Element-wise subtraction of 2 vectors where the second vector is
    subtracted from the first. Return `x1 - x2`.

    Args:
        x1 (list): First vector. 1 or 2 dimensional.
        x2 (list): Second vector. Same dimenions as `x1`.

    Returns:
        list: A element-wise subtracted list. Same dimensions as `x1`.
    """
    assert len(x1) == len(x2)

    if isinstance(x1[0], list):  # 2 dimensional
        assert len(x1[0]) == len(x2[0])
        diff = []

        for x1_i, x2_i in zip(x1, x2):
            diff_i = [x1_ij - x2_ij for x1_ij, x2_ij in zip(x1_i, x2_i)]
            diff.append(diff_i)

        return diff
    else:  # 1 dimensional
        return [x1_i - x2_i for x1_i, x2_i in zip(x1, x2)]


def transpose(W):
    """Return the transpose of a 2 dimensional list, matrix.

    Args:
        W (list): The matrix to transpose. 2 dimensional.

    Returns:
        list: Transpose of `W`.
    """
    return [list(xi) for xi in zip(*W)]


if __name__ == '__main__':
    # Run the task in Assignment 3, i.e. one forward pass followed by
    # a back propagation and an update of the parameters.

    # Create the date
    x = [1.6, -1.2, -0.21, 0.58]
    W1 = [[0.22, 1.1, 1.3, -0.12], [-0.8, 0.49, 1.2, -0.31]]
    b1 = [0.1, 0.34]
    W2 = [[0.3, 0.15], [-0.71, 1.2], [0.2, 0.98], [1.1, -0.84]]
    b2 = [-0.38, 0.74, 0.2, 0.62]
    eta = 0.01  # Learning rate

    # Forward propagation
    h, y = forward(x, W1, b1, W2, b2)

    # Compute the loss so that it can be compared to the post iteration loss
    loss_prior_iter = mean_squared_error(y, x)

    # Backward propagation
    gW1, gb1, gW2, gb2 = backward(x, h, y, W1, b1, W2, b2)

    assert len(gW1) == len(W1)
    assert len(gW1[0]) == len(x)
    assert len(gb1) == len(W1)
    assert len(gW2) == len(x)
    assert len(gW2[0]) == len(W1)
    assert len(gb2) == len(x)

    # Update parameters
    W2 = optimize_sgd(W2, gW2, eta)
    b2 = optimize_sgd(b2, gb2, eta)
    W1 = optimize_sgd(W1, gW1, eta)
    b1 = optimize_sgd(b1, gb1, eta)

    # Loss after the weight update
    h, y = forward(x, W1, b1, W2, b2)
    loss_after_iter = mean_squared_error(y, x)

    print('Loss before iteration: {}'.format(loss_prior_iter))
    print('Loss after iteration: {}'.format(loss_after_iter))

    assert loss_after_iter < loss_prior_iter
