# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import mxnet as mx
from mxnet import nd, autograd
# from pydiffpriv import cgfbank
from autodp import rdp_bank

# This is to shut pyDiffPriv up about small lambdas. TODO: Change in local pyDiffPriv code or suggest an update.
import warnings
warnings.filterwarnings("ignore")


class DpOptimizer:
    def __init__(self, hyperparams, net, params, loss_func, model_ctx, accountant):
        self._hyperparams = hyperparams

        # Store network and parameter info
        self._net = net
        self._params = params
        self._loss_func = loss_func
        self._model_ctx = model_ctx

        # Store privacy info
        self._accountant = accountant
        # self._cgf_func = lambda x: cgfbank.CGF_gaussian({'sigma': self._hyperparams['z']}, x)
        self._cgf_func = lambda x: rdp_bank.RDP_gaussian({'sigma': self._hyperparams['z']}, x)

        # Keep track of the number of steps (i.e., # of updates to the params vector)
        self._step = 0

        # Use a batch_size that fits in GPU memory
        self._batch_size = self._compute_good_batch_size()

    # Perform a single optimization step
    def step(self, data, labels, accumulate_privacy=True, run_training=True):
        lot_mean_loss = 0
        if run_training:
            # perform minimization step and update the parameters
            lot_mean_loss = self._minimize(data, labels)

        # update current amount of privacy budget consumed
        if accumulate_privacy:
            self._accountant.compose_subsampled_mechanism(self._cgf_func, self._hyperparams['sample_fraction'])

        self._step += 1

        return lot_mean_loss

    # Minimization of lot loss with batch processing
    def _minimize(self, data, labels):
        lot_loss = 0

        # Create storage for batches of summed gradients
        accumulated_grads = {}
        for param_name, param in self._params.items():
            accumulated_grads[param_name] = nd.zeros_like(param)

        for start_idx in range(0, self._hyperparams['lot_size'], self._batch_size):
            end_idx = min(self._hyperparams['lot_size'], start_idx + self._batch_size)
            batch_data = nd.slice_axis(data, axis=0, begin=start_idx, end=end_idx)
            batch_labels = nd.slice_axis(labels, axis=0, begin=start_idx, end=end_idx)
            # compute sum of clipped gradients for this batch of this lot
            lot_loss += self._accumulate_batch_gradients(batch_data, batch_labels, accumulated_grads)
            # then wait for computation to finish so that memory can be cleaned up before next batch
            nd.waitall()

        # use the computed gradients to update the parameters
        self._update_params(accumulated_grads)

        # block here, since the next step will depend on this result
        return lot_loss.asscalar() / self._hyperparams['lot_size']

    # Optimizer-specific parameter update method
    def _update_params(self, accumulated_grads):
        raise NotImplementedError("Method not implemented.")

    # Computes gradients for a single batch, adds them into the gradient accumulator
    def _accumulate_batch_gradients(self, data, labels, accumulated_grads):
        num_in_batch = data.shape[0]

        batch_params = self._get_batch_params(num_in_batch)
        with autograd.record():
            output = self._net(data, batch_params, training_mode=True)
            loss = self._loss_func(output, labels)
        loss.backward()

        px_clipping_factors = self._compute_px_clipping_factors(batch_params, num_in_batch)

        for param_name, param in self._params.items():
            batch_param_grad = batch_params[param_name].grad
            # clip the gradients of each example using the computed clipping factors
            clipped_batch_grads = self._clip_px_gradients(batch_param_grad, px_clipping_factors)
            # sum the gradients up along the batch axis, then store the result in the accumulator
            accumulated_grads[param_name] += nd.sum(clipped_batch_grads, axis=0)

        return nd.sum(loss)

    # Manually broadcast params by adding a dimension of lot_size size, that we can get per-example gradients
    def _get_batch_params(self, num_in_batch):
        batch_params = {}
        for param_name, param in self._params.items():
            batch_param = param.broadcast_to((num_in_batch,) + param.shape)
            batch_param.attach_grad()
            batch_params[param_name] = batch_param
        return batch_params

    # Compute how much the gradients of each example should be clipped
    def _compute_px_clipping_factors(self, batch_params, num_in_batch):
        px_norms = self._compute_px_gradient_norms(batch_params, num_in_batch)
        l2_rescales = self._hyperparams['l2_clipping_bound'] / (px_norms + 1e-8)  # tiny additive term to prevent div_by_0
        one_vs_rescale = nd.stack(nd.ones_like(px_norms, ctx=self._model_ctx), l2_rescales, axis=1)
        return nd.min(one_vs_rescale, axis=0, exclude=True)

    # Compute norms of gradients for each example in the batch
    def _compute_px_gradient_norms(self, batch_params, num_in_batch):
        batch_norms = nd.zeros(shape=(num_in_batch), ctx=self._model_ctx)
        for param_name, batch_param in batch_params.items():
            batch_norms += nd.sum(nd.square(batch_param.grad), axis=0, exclude=True)
        return nd.sqrt(batch_norms)

    def _clip_px_gradients(self, batch_grads, px_clipping_factors):
        # hacky workaround for not knowing how to multiply a (b,) shape array with a (b, x) or (b, x, y) shape array
        expanded_batch_clipping_factors = nd.expand_dims(px_clipping_factors, 1)
        if len(batch_grads.shape) == 3:
            expanded_batch_clipping_factors = nd.expand_dims(expanded_batch_clipping_factors, 1)
        return nd.multiply(batch_grads, expanded_batch_clipping_factors)

    # Rough estimate for the largest batch size <= lot_size that will fit in 16GB of GPU memory and yield fairly even-size batches
    def _compute_good_batch_size(self):
        GB_LIMIT = 16.
        FUDGE_FACTOR = 2.71
        PARAM_SIZE_IN_GB = 64. / (8 * 1024 * 1024 * 1024)
        num_params = sum([param.size for param in self._params.values()])
        px_gb_estimate = FUDGE_FACTOR * PARAM_SIZE_IN_GB * num_params
        # whats the largest batch size B such that B * px_gb_estimate < 16?
        largest_batch = GB_LIMIT / px_gb_estimate
        if self._hyperparams['lot_size'] <= largest_batch:
            return self._hyperparams['lot_size']
        # what if this largest_batch size leaves a very uneven final batch?
        num_batches = math.ceil(self._hyperparams['lot_size'] / largest_batch)
        return math.ceil(self._hyperparams['lot_size'] / num_batches)
