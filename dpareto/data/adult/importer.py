# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from mxnet import nd


def import_adult_dataset(data_ctx):
    try:
        #print("Unpickling...")
        Xtrain, Ytrain = pickle.load( open( "dpareto/data/adult/a1a.train.p", "rb" ) )
        Xtest, Ytest = pickle.load( open( "dpareto/data/adult/a1a.test.p", "rb" ) )
        #print("unpickled.")
        return Xtrain, Ytrain, Xtest, Ytest
    except FileNotFoundError as _:
        print("unpickling failed.")
        print("Opening...")
        with open("dpareto/data/adult/a1a.train") as f:
            train_raw = f.read()

        with open("dpareto/data/adult/a1a.test") as f:
            test_raw = f.read()
        print("done opening.")

        def process_data(raw_data):
            train_lines = raw_data.splitlines()
            num_examples = len(train_lines)
            num_features = 123
            X = nd.zeros((num_examples, num_features), ctx=data_ctx)
            Y = nd.zeros((num_examples, 1), ctx=data_ctx)
            for i, line in enumerate(train_lines):
                tokens = line.split()
                label = (int(tokens[0]) + 1) / 2  # Change label from {-1,1} to {0,1}
                Y[i] = label
                for token in tokens[1:]:
                    index = int(token[:-2]) - 1
                    X[i, index] = 1
            return X, Y

        print("Processing...")
        Xtrain, Ytrain = process_data(train_raw)
        Xtest, Ytest = process_data(test_raw)
        print("done processing.")

        make_pickle = False
        if make_pickle:
            print("Pickling...")
            pickle.dump( (Xtrain, Ytrain), open( "dpareto/data/adult/a1a.train.p", "wb" ) )
            pickle.dump( (Xtest, Ytest), open( "dpareto/data/adult/a1a.test.p", "wb" ) )
            print("pickled.")
        return Xtrain, Xtest, Ytrain, Ytest
