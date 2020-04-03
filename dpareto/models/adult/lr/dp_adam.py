# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mxnet as mx
from mxnet import nd, gluon
import numpy as np

from dpareto.models.adult.lr.base import AdultLrBase


class DpAdultLrAdam(AdultLrBase):
    def __init__(self, hyperparams, options={}):
        options['optimizer_name'] = 'dp_adam'
        super(DpAdultLrAdam, self).__init__(hyperparams, options)

        self._input_layer = 123
        self._hidden_layers = []
        self._output_layer = 1

        self._name = 'dp_adult_lr_adam'


# Small run for testing purposes.
def main():
    print("Running dp_adult_lr_adam's main().")

    # Fixing some combo of these random seeds is useful for debugging. I'll fix them all to be safe.
    import random
    random_seed = 112358
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)

    # Some default hyperparameter setting.
    hyperparams = {'epochs': 1,
                   'lot_size': 64,
                   'lr': 0.05,
                   'l2_clipping_bound': 2.0,
                   'z': 1.0,
                   'fixed_delta': 1e-5,
                   'beta_1': 0.9,
                   'beta_2': 0.999}
    print("hyperparams: {}".format(hyperparams))

    options = {'use_gpu': False, 'verbose': True, 'accumulate_privacy': True, 'debugging': True}

    instance = DpAdultLrAdam(hyperparams, options)
    priv, acc = instance.run()
    print("Instance privacy: {}".format((priv, hyperparams['fixed_delta'])))
    print("Instance accuracy: {}".format(acc))


if __name__ == "__main__":
    main()
