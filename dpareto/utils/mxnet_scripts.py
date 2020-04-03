# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mxnet as mx


def get_gpu_count():
    max_gpus = 64
    for i in range(max_gpus):
        try:
            mx.nd.zeros((1,), ctx=mx.gpu(i))
        except:
            return i
    return max_gpus