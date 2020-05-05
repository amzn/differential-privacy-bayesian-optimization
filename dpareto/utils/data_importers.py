# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle


def import_adult_dataset(data_path, data_ctx):
    try:
        Xtrain, Ytrain = pickle.load(open(os.path.join(data_path, "adult/a1a.train.p"), "rb"))
        Xtest, Ytest = pickle.load(open(os.path.join(data_path, "adult/a1a.test.p"), "rb"))
        return Xtrain, Ytrain, Xtest, Ytest
    except FileNotFoundError:
        print(f'Adult dataset files not found in {data_path}. Try running the downloader from the project root.')
