# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from mxnet import nd


def import_adult_dataset(data_ctx):
    try:
        Xtrain, Ytrain = pickle.load( open( "dpareto/data/adult/a1a.train.p", "rb" ) )
        Xtest, Ytest = pickle.load( open( "dpareto/data/adult/a1a.test.p", "rb" ) )
        return Xtrain, Ytrain, Xtest, Ytest
    except FileNotFoundError:
        print('Adult dataset files not found. Try running the downloader from the project root.')
