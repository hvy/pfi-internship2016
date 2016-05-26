import time
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers
from utils import dataset, exporter


class Autoencoder(chainer.Chain):
    def __init__(self):
        # Test different number of hidden units and layers.
        super().__init__(
            l1=L.Linear(10, 5),
            l2=L.Linear(5, 10)
        )

    def __call__(self, x, t):
        # Testing different activation functions.
        h = self.l1(x)
        y = self.l2(h)

        # Scale the MSE by 5, i.e  0.5 * 10 so that the loss can be compared to
        # the loss computed in Assignment 4. Factor 0.5, since the Chainer
        # implementation doesn't scale the error by 0.5 and factor 10, since
        # the previous assignment loss functions does not compute the mean,
        # and the number of summed elements are 10.
        self.loss = 5 * F.mean_squared_error(y, t)

        return self.loss


if __name__ == '__main__':
    # Train an Autoencoder with with various configurations to observe how
    # the loss changes over the 1000 epochs.

    N, D, xs = dataset.read('data/dataset.dat')
    assert len(xs) == N

    n_epochs = 100000

    # Test different batch sizes.
    batchsize = 100

    # Test different optimization algoritms.
    optimizer = optimizers.SGD()
    # optimizer = optimizers.MomentumSGD()
    # optimizer = optimizers.AdaGrad()
    # optimizer = optimizers.Adam()
    # optimizer = optimizers.RMSprop()
    # optimizer = optimizers.RMSpropGraves()

    model = Autoencoder()
    optimizer.setup(model)

    xs = np.asarray(xs, dtype=np.float32)

    losses = []

    noise = True

    x_all = Variable(np.asarray(xs), volatile=True)
    t_all = Variable(np.asarray(xs), volatile=True)

    for epoch in range(n_epochs):

        # volatile = 'off' if model.train else 'on'

        total_loss = 0

        indexes = np.random.permutation(N)

        for i in range(0, N, batchsize):

            x = Variable(np.asarray(xs[indexes[i:i + batchsize]]))
            t = Variable(np.asarray(xs[indexes[i:i + batchsize]]))

            optimizer.update(model, x, t)

        average_loss = model(x_all, t_all)
        # average_loss = total_loss / N
        print('Loss: {}'.format(average_loss.data))


    # Uncomment the following lines to save the trained parameters to a file
    # out_filename = 'output/assignment5_loss_' + str(int(time.time()) + '.csv')
    # out_filename = 'output/assignment5_loss_' + str(batchsize) + '.csv'
    # exporter.export_list(out_filename, losses)
