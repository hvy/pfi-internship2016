def export_list(filename, data, delimiter='\n'):
    """Export data (a list) with the given filename where each element
    in the list is separated by a delimiter.

    Args:
        filename (str): Path of the file.
        data (list): Data to export. 1 dimensional.
        delimiter (str): Delimiter to insert between each data entry.
    """
    with open(filename, 'w') as out:
        out.write(delimiter.join(map(str, data)))


def export_model(filename, model, delimiter=' '):
    """Export a model with the given filename according to the format
    speficied by the instructions.

    Args:
        filename (str): Path of the file.
        model (dict): Model with the parameters `W1`, `b1`, `W2`, `b2`.
        delimiter (str): Delimiter to insert between each parameter value.
    """
    W1 = model.W1
    W2 = model.W2
    b1 = model.b1
    b2 = model.b2

    with open(filename, 'w') as out:
        for W_i in W1:
            out.write(delimiter.join(map(str, W_i)) + '\n')
        for W_i in W2:
            out.write(delimiter.join(map(str, W_i)) + '\n')

        out.write(delimiter.join(map(str, b1)) + '\n')
        out.write(delimiter.join(map(str, b2)) + '\n')
