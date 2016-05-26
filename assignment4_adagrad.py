import math
import time
from utils import dataset, randomizer, exporter
from assignment3 import optimize_sgd
from assignment4 import Autoencoder


class AutoencoderAdaGrad(Autoencoder):
    """Simple Autoencoder implementation with one hidden layer, that uses the
    identity function f(x) = x as its activation function and optimizes with
    Stochastic Gradient Descent (SGD).

    The learning rates for each parameters is adapted individually using
    AdaGrad.
    """
    def __init__(self, n_in, n_units, lr=0.01, mean=0, stddev=0.01):
        """Model constructor, initializing the parameters.

        Args:
            n_in (int):  Number of input units, i.e. size of the input vector.
            n_units (int): Number of hidden units.
            lr (float): Initial learning rate.
            mean (float): Mean of the initial parameters.
            stddev (float): Standard deviation of the initial random
                parameters.
        """
        super().__init__(n_in, n_units, lr, mean, stddev)

        # AdaGrad gradient history initialization.
        self.eps = 1e-8  # Smoothing term.
        self.gW1_history = [[0] * n_in for _ in range(n_units)]
        self.gb1_history = [0] * n_units
        self.gW2_history = [[0] * n_units for _ in range(n_in)]
        self.gb2_history = [0] * n_in

    def optimize(self, h, y, gW1, gb1, gW2, gb2):
        """Optimizes (modifies) the parameters of this model using AdaGrad
        and SGD.

        Args:
            h (list): Activations of the hidden layer.
            y (list): Activations of the output layer.
            gW1 (list): Computed gradients of `W1`.
            gb1 (list): Computed gradients of `b1`.
            gW2 (list): Computed gradients of `W2`.
            gb2 (list): Computed gradients of `bw`.
        """
        # Update gradient history.
        gW2, self.gW2_history = self.adagrad_iter(gW2, self.gW2_history,
                                                  self.eps)
        gb2, self.gb2_history = self.adagrad_iter(gb2, self.gb2_history,
                                                  self.eps)
        gW1, self.gW1_history = self.adagrad_iter(gW1, self.gW1_history,
                                                  self.eps)
        gb1, self.gb1_history = self.adagrad_iter(gb1, self.gb1_history,
                                                  self.eps)

        self.W2 = optimize_sgd(self.W2, gW2, self.lr)
        self.b2 = optimize_sgd(self.b2, gb2, self.lr)
        self.W1 = optimize_sgd(self.W1, gW1, self.lr)
        self.b1 = optimize_sgd(self.b1, gb1, self.lr)

    def adagrad_iter(self, g, g_history, eps=1e-8):
        """Helper method for AdaGrad. Update history, then compute the new
        gradients.

        Args:
            g (list): The newly computed gradients. 1 or 2 dimensional.
            g_history (list): The accumulated history of the grdients. Same
                dimensions as `g`.
            eps (float): Smoothing term. Usually in the range 1e-4 to 1e-8.

        Returns:
            list: The adapted newly computed gradients.
            list: The updated gradient history.
        """
        assert len(g) == len(g_history)

        if isinstance(g[0], list):  # 2 dimensional
            assert len(g[0]) == len(g_history[0])
            for i in range(len(g)):
                for j in range(len(g[i])):
                    g_history[i][j] += g[i][j] ** 2
                    g[i][j] /= eps + math.sqrt(g_history[i][j])

        else:  # 1 dimensional
            for i in range(len(g)):
                g_history[i] += g[i] ** 2
                g[i] /= eps + math.sqrt(g_history[i])

        return g, g_history


if __name__ == '__main__':
    # Train an Autoencoder model, with AdaGrad

    N, D, xs = dataset.read('data/dataset.dat')

    mean = 0
    stddev = math.sqrt(1 / D)

    n_hidden_units = 5
    n_epochs = 20

    # The initial learning rate should be much higher e.g. 1.0 than in normal
    # SGD without AdaGrad, since the adaption will take care of the scaling.
    initial_learning_rate = 1

    model = AutoencoderAdaGrad(n_in=D, n_units=n_hidden_units,
                               lr=initial_learning_rate, mean=mean,
                               stddev=stddev)

    for epoch in range(n_epochs):
        randomizer.shuffle(xs)

        for x in xs:
            model(x, train=True)

        total_loss = 0
        for x in xs:
            loss = model(x, train=False)
            total_loss += loss
        average_loss = total_loss / N

        print('Epoch: {} Avg. loss: {}'.format(epoch, average_loss))

    # out_filename = 'output/assignment4_AdaGrad_params_' + str(int(time.time()))
    # exporter.export_model(out_filename, model)
