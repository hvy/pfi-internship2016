import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers
from utils import dataset


class Autoencoder(chainer.Chain):
    def __init__(self):
        # Test different number of hidden units and layers.
        super(Autoencoder, self).__init__(
            l1=L.Linear(10, 5),
            l2=L.Linear(5, 10)
        )

    def __call__(self, x):
        # Testing different activation functions.
        h = self.l1(x)
        y = self.l2(h)

        self.loss = F.mean_squared_error(y, x)

        return self.loss


if __name__ == '__main__':
    # Train an Autoencoder with with various configurations to observe how
    # the loss changes over the 1000 epochs.

    N, D, xs = dataset.read('dataset.dat')
    n_epochs = 1000

    # Test different batch sizes.
    batchsize = 1

    # Test different optimization algoritms.
    optimizer = optimizers.SGD()
    # optimizer = optimizers.MomentumSGD()
    # optimizer = optimizers.AdaGrad()

    model = Autoencoder()
    optimizer.setup(model)

    xs = np.asarray(xs, dtype=np.float32)

    for epoch in range(n_epochs):

        trainsize = N
        indexes = np.random.permutation(trainsize)

        for i in range(0, trainsize, batchsize):
            x = Variable(np.asarray(xs[indexes[i:i + batchsize]]))
            optimizer.update(model, x)

        xs_var = Variable(xs)
        loss = model(xs_var)

        print('Epoch: {} Avg. loss: {}'.format(i, loss.data))

    print('done')
