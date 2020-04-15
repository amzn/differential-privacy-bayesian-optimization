# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import mxnet as mx
from mxnet import nd
import urllib.request
import shutil
import ssl

print("Downloading...")
training_data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t'
testing_data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a'
context = ssl._create_unverified_context()  # Bypasses failing HTTPS verification for the dataset site
training_data = urllib.request.urlopen(training_data_url, context=context).read().decode('utf-8')
testing_data = urllib.request.urlopen(testing_data_url, context=context).read().decode('utf-8')
print("downloaded.")

def process_data(raw_data):
    data_ctx = mx.cpu()
    lines = raw_data.splitlines()
    num_examples = len(lines)
    num_features = 123
    X = nd.zeros((num_examples, num_features), ctx=data_ctx)
    Y = nd.zeros((num_examples, 1), ctx=data_ctx)
    for i, line in enumerate(lines):
        tokens = line.split()
        label = (int(tokens[0]) + 1) / 2  # Change label from {-1,1} to {0,1}
        Y[i] = label
        for token in tokens[1:]:
            index = int(token[:-2]) - 1
            X[i, index] = 1
    return X, Y

print("Processing...")
Xtrain, Ytrain = process_data(training_data)
Xtest, Ytest = process_data(testing_data)
print("processed.")

print("Pickling...")
pickle.dump( (Xtrain, Ytrain), open( "dpareto/data/adult/a1a.train.p", "wb" ) )
pickle.dump( (Xtest, Ytest), open( "dpareto/data/adult/a1a.test.p", "wb" ) )
print("pickled.")
