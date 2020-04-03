# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mxnet as mx
from mxnet import nd

from dpareto.optimizers.dp_optimizer import DpOptimizer

class DpAdam(DpOptimizer):
    def __init__(self, hyperparams, net, params, loss_func, model_ctx, accountant):
        super(DpAdam, self).__init__(hyperparams, net, params, loss_func, model_ctx, accountant)

        # Compute scale of Gaussian noise to add
        self._hyperparams['sigma'] = hyperparams['z'] * (2 * hyperparams['l2_clipping_bound'] / hyperparams['lot_size'])

        # Initialize 1st and 2nd moment vectors
        self._m = {}
        self._v = {}
        for param_name, param in self._params.items():
            self._m[param_name] = nd.zeros_like(param)
            self._v[param_name] = nd.zeros_like(param)

    def _update_params(self, accumulated_grads):
        # scale gradients by lot size, add noise, and update the parameters
        for param_name, param in self._params.items():
            # average the clipped gradients and then add noise to each averaged gradient
            param_grad_update = (accumulated_grads[param_name] / self._hyperparams['lot_size']) + \
                                mx.random.normal(0, self._hyperparams['sigma'], param.shape, ctx=self._model_ctx)

            # update biased first moment estimate
            self._m[param_name] = self._hyperparams['beta_1'] * self._m[param_name] + (1 - self._hyperparams['beta_1']) * param_grad_update

            # update biased second raw moment estimate
            self._v[param_name] = self._hyperparams['beta_2'] * self._v[param_name] + (1 - self._hyperparams['beta_2']) * nd.square(param_grad_update)

            # compute bias-corrected first moment estimate
            m_hat = self._m[param_name] / (1 - nd.power(self._hyperparams['beta_1'], self._step + 1))

            # compute bias-corrected second raw moment estimate
            v_hat = self._v[param_name] / (1 - nd.power(self._hyperparams['beta_2'], self._step + 1))

            # update params with ADAM
            param[:] = param - self._hyperparams['lr'] * m_hat / (nd.sqrt(v_hat) + 1e-8)
