# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import mxnet as mx
import numpy as np


def set_random_seed(value):
    wrapped_value =  value % (2**32 - 1)
    random.seed(wrapped_value)
    np.random.seed(wrapped_value)
    mx.random.seed(wrapped_value)