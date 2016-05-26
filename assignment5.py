import time
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers
from utils import dataset, exporter


class Autoencoder(chainer.Chain):
    def __init__(self):
        """Default constructor, defining the connections of an Autoencoder."""
        # Test different number of hidden units and layers.
        super().__init__(
            l1=L.Linear(10, 5),
            l2=L.Linear(5, 10)
        )
        self.train = True

    def __call__(self, x, t):
        """Perform a forward pass and compute the loss. This method ultimately
        defines the model.

        Args:
            x (chainer.Variable): Input vector.
            t (chainer.Variable): Target vector. Usually identical to `x` in
                the case of an Autoencoder.

        Returns:
            chainer.Variable: Loss.
        """
        # Test different activation functions and dropout.
        h = self.l1(x)
        y = self.l2(h)

        if self.train:
            # Scale the MSE by 5, i.e  0.5 * 10 so that the loss can be compared to
            # the loss computed in Assignment 4. Factor 0.5, since the Chainer
            # implementation doesn't scale the error by 0.5 and factor 10, since
            # the previous assignment loss functions does not compute the mean,
            # and the number of summed elements are 10.
            self.loss = 5 * F.mean_squared_error(y, t)

            return self.loss
        else:
            return y


def split(xs, fst_half_rate=0.7):
    """Split the given array in two with a certain ratio.

    Args:
        xs (numpy.ndarray): Array to split into two.
        fst_half_rate (float): Rate deciding the length of the first split.
    """
    train, test = np.split(xs.copy(), [int(xs.shape[0] * fst_half_rate)])
    return train, test


def add_masking_noise(batch, rate=0.3):
    """Add masking noise to the given batch, i.e. set random elements in the
    bactch to 0. Note that this method modifies the original array.

    Args:
        batch (numpy.ndarray): Array to be masked.
        rate (float): Rate deciding how many elements will be set to 0.
    """
    D = batch.shape[1]
    for x in batch:
        masks = np.random.permutation(D)[:int(D * rate)]
        x[masks] = 0


if __name__ == '__main__':
    # Train an Autoencoder with with various configurations to observe how
    # the loss changes over time.

    N, D, xs = dataset.read('data/dataset.dat')

    n_epochs = 1000

    # Test different batch sizes.
    batchsize = 100

    # Proprocess the training data, e.g. split it for cross validation
    # or add noise to train a Denoising Autoencoder

    xs = np.asarray(xs, dtype=np.float32)

    # Use a subset of the training data as the validation data.
    cross_validation_rate = 0  # 0 disables cross validation
    if cross_validation_rate:
        xs_train, xs_val = split(xs, fst_half_rate=cross_validation_rate)
    else:
        xs_train = xs.copy()
        xs_val = xs.copy()

    # Use a copy of the input data as the target data, even though they are
    # identical, in case of noise addition to the input
    x_train = xs_train.copy()
    y_train = xs_train.copy()

    # Train a Denoising Autoencoder
    noise_rate = 0  # 0 to add no noise
    if noise_rate:
        add_masking_noise(x_train, rate=noise_rate)

    # Test different optimization algoritms.
    # optimizer = optimizers.SGD(lr=0.01)
    # optimizer = optimizers.AdaGrad(lr=0.001, eps=1e-08)
    # optimizer = optimizers.AdaDelta(rho=0.95, eps=1e-06)
    # optimizer = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
    # optimizer = optimizers.RMSprop(lr=0.01, alpha=0.99, eps=1e-08)
    # optimizer = optimizers.RMSpropGraves(lr=0.0001, alpha=0.95, momentum=0.9, eps=0.0001)

    model = Autoencoder()
    optimizer.setup(model)

    # Store epoch losses for plotting purpose
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):

        trainsize = x_train.shape[0]
        indexes = np.random.permutation(trainsize)

        # Train
        for i in range(0, trainsize, batchsize):
            x = Variable(np.asarray(x_train[indexes[i:i + batchsize]]))
            t = Variable(np.asarray(y_train[indexes[i:i + batchsize]]))
            optimizer.update(model, x, t)

        # Loss on training data
        x = Variable(np.asarray(x_train), volatile=True)
        t = Variable(np.asarray(y_train), volatile=True)
        train_loss = model(x, t)
        train_losses.append(train_loss.data)

        if cross_validation_rate > 0:
            # Loss on validation data
            x = Variable(np.asarray(xs_val), volatile=True)
            val_loss = model(x, x)
            val_losses.append(val_loss.data)

        print('Epoch: {} Loss(Training): {}'.format(epoch, train_loss.data))

    # Uncomment the following lines to save the trained parameters to a file
    # exporter.export_list('output/val_losses.csv', val_losses)
    # exporter.export_list('output/train_losses.csv', val_losses)
