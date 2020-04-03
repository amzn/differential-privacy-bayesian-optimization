# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mxnet as mx
from mxnet import nd, gluon
import numpy as np

from dpareto.models.mnist.base import MnistBase


class DpMnistAdam(MnistBase):
    def __init__(self, hyperparams, options={}):
        options['optimizer_name'] = 'dp_adam'
        super(DpMnistAdam, self).__init__(hyperparams, options)

        self._input_layer = 28*28
        self._hidden_layers = options['hidden_layers']
        self._output_layer = 10

        self._name = 'dp_mnist_adam'


# Small run for testing purposes.
def main():
    print("Running dp_mnist_adam's main().")

    # Fixing some combo of these random seeds is useful for debugging. I'll fix them all to be safe.
    import random
    random_seed = 112358
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)

    # Set hyperparams as default from Google's "Deep Learning with DP" paper.
    hyperparams = {'epochs': 1,
                   'lot_size': 600,
                   'lr': 0.05,
                   'l2_clipping_bound': 4.0,
                   'z': 4.0,
                   'fixed_delta': 1e-5,
                   'beta_1': 0.9,
                   'beta_2': 0.999}
    print("hyperparams: {}".format(hyperparams))

    options = {'use_gpu': True, 'verbose': True, 'hidden_layers': [1000], 'debugging': True}

    priv, acc = DpMnistAdam(hyperparams, options).run()
    print("Instance privacy: {}".format((priv, hyperparams['fixed_delta'])))
    print("Instance accuracy: {}".format(acc))

    import gc
    gc.collect()  # necessary to avoid out-of-memory errors on sequential runs


if __name__ == "__main__":
    main()
