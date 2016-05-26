import math
import time
from utils import dataset, randomizer, exporter
from assignment3 import forward, backward, squared_error, optimize_sgd


class Autoencoder:
    """Simple Autoencoder implementation with one hidden layer, that uses the
    identity function f(x) = x as its activation function and optimizes with
    Stochastic Gradient Descent (SGD).
    """
    def __init__(self, n_in, n_units, lr=0.01, mean=0.0, stddev=0.01):
        """Model constructor, initializing the parameters.

        Args:
            n_in (int):  Number of input units, i.e. size of the input vector.
            n_units (int): Number of hidden units.
            lr (float): Initial learning rate.
            mean (float): Mean of the initial parameters.
            stddev (float): Standard deviation of the initial random
                parameters.
        """
        self.W1 = randomizer.rnd((n_units, n_in), mean, stddev)
        self.b1 = randomizer.rnd((n_units,), mean, stddev)
        self.W2 = randomizer.rnd((n_in, n_units), mean, stddev)
        self.b2 = randomizer.rnd((n_in,), mean, stddev)
        self.lr = lr

    def __call__(self, x, train=True):
        """Perform an iteration with one sample, updating the parameters of
        this model if the training flag is set to true, otherwise just
        perform a forward propagation and return the loss.

        Args:
            x (list): Input data. 1 dimensional.
            train (bool): True if the parameters are to be updated. False
                otherwise.

        Returns:
            float: Loss.
        """
        h, y = forward(x, self.W1, self.b1, self.W2, self.b2)

        if train:
            gW1, gb1, gW2, gb2 = backward(x, h, y, self.W1, self.b1, self.W2,
                                          self.b2)
            self.optimize(h, y, gW1, gb1, gW2, gb2)

        loss = squared_error(y, x, scale=0.5)

        return loss

    def optimize(self, h, y, gW1, gb1, gW2, gb2):
        """Optimizes (modifies) the parameters of this model using SGD.

        Args:
            h (list): Activations of the hidden layer.
            y (list): Activations of the output layer.
            gW1 (list): Computed gradients of `W1`.
            gb1 (list): Computed gradients of `b1`.
            gW2 (list): Computed gradients of `W2`.
            gb2 (list): Computed gradients of `bw`.
        """
        self.W2 = optimize_sgd(self.W2, gW2, self.lr)
        self.b2 = optimize_sgd(self.b2, gb2, self.lr)
        self.W1 = optimize_sgd(self.W1, gW1, self.lr)
        self.b1 = optimize_sgd(self.b1, gb1, self.lr)


if __name__ == '__main__':
    # Train an Autoencoder model

    N, D, xs = dataset.read('data/dataset.dat')

    # Parameters for initializing the ranom parameters. The standard deviation
    # is set with respect to the dimensions of the inputs.
    # http://docs.chainer.org/en/stable/reference/links.html#linear
    mean = 0
    stddev = math.sqrt(1 / D)

    n_hidden_units = 5
    n_epochs = 20
    initial_learning_rate = 0.001

    model = Autoencoder(n_in=D, n_units=n_hidden_units,
                        lr=initial_learning_rate, mean=mean, stddev=stddev)

    # Start the training
    for epoch in range(n_epochs):
        randomizer.shuffle(xs)

        # Optimize the model
        for x in xs:
            model(x, train=True)

        # Compute the loss
        total_loss = 0
        for x in xs:
            loss = model(x, train=False)
            total_loss += loss
        average_loss = total_loss / N

        print('Epoch: {} Avg. loss: {}'.format(epoch + 1, average_loss))

    # Uncomment the following lines to save the trained parameters to a file
    # out_filename = 'output/new_assignment4_params_' + str(int(time.time()))
    # exporter.export_model(out_filename, model)
