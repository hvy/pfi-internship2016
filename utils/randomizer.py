import random


def rnd(shape, mean=0.0, stddev=0.01):
    """Return a list with the given shape with random float elements.
    The values are generated from a Gaussian distribution.

    Args:
        shape (tuple, list): A shape to generate random floats for. 1 or 2 dimensional.
        mean (float): Mean of the Gaussian distribution.
        stddev (float): Standard deviation of the distribution.

    Returns:
        list: A list with random float elements with the dimensions specified by `shape`.
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError('Invalid shape. The shape must be a tuple or a list.')

    dimension = len(shape)

    if dimension is 1:
        # Create a 1 dimensional array with random elements
        return [random.gauss(mean, stddev) for _ in range(shape[0])]
    elif dimension is 2:
        # Create a 2 dimensional matrix with random elements
        return [[random.gauss(mean, stddev) for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise NotImplementedError('Random generation with dimensions > 2 is not yet implemented.')


def shuffle(x):
    """Shuffle (modify) the given array. If the list dimensionality is higher
    than one, the elements in the higher dimensions are unchanged.

    Refer to the `random` implementation.

    Args:
        x (list): A list to shuffle.
    """
    random.shuffle(x)
