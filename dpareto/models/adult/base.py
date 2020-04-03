# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mxnet as mx
from mxnet import nd, gluon
import numpy as np

import dpareto.data.adult.importer as importer
from dpareto.models.dp_feedforward_net import DpFeedforwardNet
from dpareto.utils.lot_sampler import LotSampler


class AdultBase(DpFeedforwardNet):
    def __init__(self, hyperparams, options={}):
        super(AdultBase, self).__init__(hyperparams, options)

    def _load_data(self):
        xTrain, yTrain, xTest, yTest = importer.import_adult_dataset(self._data_ctx)

        num_training_examples = len(yTrain)
        num_testing_examples = len(yTest)

        if self._verbose:
            print("Loading...")
        train_data = gluon.data.ArrayDataset(xTrain, yTrain)
        train_data_lot_iterator = gluon.data.DataLoader(train_data,
                                                        batch_sampler=LotSampler(self._lot_size, num_training_examples))
        train_data_eval_iterator = gluon.data.DataLoader(train_data, batch_size=self._lot_size)
        test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(xTest, yTest),
                                          batch_size=self._lot_size, shuffle=True)
        if self._verbose:
            print("done loading.")

        return num_training_examples, num_testing_examples, train_data_lot_iterator, train_data_eval_iterator, test_data

    def _evaluate_accuracy(self, data_iterator):
        num_correct = 0
        total = 0
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(self._model_ctx).reshape((-1, 1, self._input_layer))
            label = label.as_in_context(self._model_ctx)

            output = self._net(data, self._params, training_mode=False)
            prediction = output > 0.0

            num_correct += nd.sum(prediction == label)
            total += len(label)

        return num_correct.asscalar() / total
