# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mxnet as mx

from dpareto.models.adult.base import AdultBase


class AdultLrBase(AdultBase):
    def __init__(self, hyperparams, options={}):
        super(AdultLrBase, self).__init__(hyperparams, options)

    @staticmethod
    def _get_loss_func():
        return mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
