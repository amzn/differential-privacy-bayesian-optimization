# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mxnet as mx
from mxnet import nd

from dpareto.optimizers.dp_optimizer import DpOptimizer

class DpSgd(DpOptimizer):
    def __init__(self, hyperparams, net, params, loss_func, model_ctx, accountant):
        super(DpSgd, self).__init__(hyperparams, net, params, loss_func, model_ctx, accountant)

        # Compute scale of Gaussian noise to add
        self._hyperparams['sigma'] = hyperparams['z'] * (2 * hyperparams['l2_clipping_bound'] / hyperparams['lot_size'])

    def _update_params(self, accumulated_grads):
        # scale gradients by lot size, add noise, and update the parameters
        for param_name, param in self._params.items():
            # average the clipped gradients and then add noise to each averaged gradient
            param_grad_update = (accumulated_grads[param_name] / self._hyperparams['lot_size']) + \
                                mx.random.normal(0, self._hyperparams['sigma'], param.shape, ctx=self._model_ctx)

            # update params with SGD
            param[:] = nd.sgd_update(weight=param, grad=param_grad_update, lr=self._hyperparams['lr'])
