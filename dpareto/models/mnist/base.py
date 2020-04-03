# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mxnet as mx
from mxnet import nd, gluon
import numpy as np

from dpareto.models.dp_feedforward_net import DpFeedforwardNet
from dpareto.utils.lot_sampler import LotSampler


class MnistBase(DpFeedforwardNet):
    def __init__(self, hyperparams, options={}):
        super(MnistBase, self).__init__(hyperparams, options)

    def _load_data(self):
        num_training_examples = 60000
        num_testing_examples = 10000

        def transform(data, label):
            return data.astype(np.float32)/255, label.astype(np.float32)

        if self._verbose:
            print("Loading...")
        train_data = mx.gluon.data.vision.MNIST(train=True, transform=transform)
        train_data_lot_iterator = gluon.data.DataLoader(train_data,
                                                        batch_sampler=LotSampler(self._lot_size, num_training_examples))
        train_data_eval_iterator = gluon.data.DataLoader(train_data, self._lot_size)
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), self._lot_size)
        if self._verbose:
            print("done loading.")

        return num_training_examples, num_testing_examples, train_data_lot_iterator, train_data_eval_iterator, test_data

    def _get_loss_func(self):
        return mx.gluon.loss.SoftmaxCrossEntropyLoss()

    def _evaluate_accuracy(self, data_iterator):
        numerator = 0.
        denominator = 0.
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(self._model_ctx).reshape((-1, 1, self._input_layer))
            label = label.as_in_context(self._model_ctx)

            output = self._net(data, self._params, training_mode=False)
            predictions = nd.argmax(output, axis=1)

            correct_in_batch = nd.sum(predictions == label)
            total_in_batch = data.shape[0]
            numerator += correct_in_batch
            denominator += total_in_batch
        return (numerator / denominator).asscalar()
