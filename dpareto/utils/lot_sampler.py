# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from mxnet.gluon.data import sampler
import numpy as np


class LotSampler(sampler.BatchSampler):
    def __init__(self, lot_size, data_size):
        self._lot_size = lot_size
        self._data_size = data_size
        super().__init__(self.RandomBatchSampler(lot_size, data_size), lot_size)

    class RandomBatchSampler(sampler.Sampler):
        """Samples a subset of elements randomly without replacement from the full dataset.
        Parameters
        ----------
        num_total : int
            Number of elements total to sample from.
        num_sample : int
            Number of elements to be sampled.
        """
        def __init__(self, num_sample, num_total):
            self._num_sample = num_sample
            self._num_total = num_total

        def __iter__(self):
            indices = np.arange(self._num_total)
            np.random.shuffle(indices)
            indices = indices[:self._num_sample]
            return iter(indices)

        def __len__(self):
            return self._num_sample
