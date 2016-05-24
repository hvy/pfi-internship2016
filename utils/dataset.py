import csv


def read(filename):
    """Read the dataset file formatted according to the 2016 PFI/PFN Internship
    assignment and return the list of rows.

    Args:
        filename (str): Path to the file.

    Returns:
        int: Number of rows in the dataset file.
        int: Number of components in each row. Dimensions.
        list: Rows of the dataset file where each row is represented by a list.
    """
    data = []

    with open(filename, 'r') as ssvfile:  # ssv for space separated.
        reader = csv.reader(ssvfile, delimiter=' ')
        for row in reader:
            data.append(row)

    # The first line of the file is expected to contain meta data.
    meta_data = data.pop(0)

    N = int(meta_data[0])  # Number of samples.
    D = int(meta_data[1])  # Sample dimension.

    # Parse to floats.
    xs = [list(map(float, x)) for x in data]

    assert N == len(xs)
    for x in xs:
        assert D == len(x)

    return N, D, xs
