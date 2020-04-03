# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import sys


class Dummy:
    def __init__(self, *args):
        pass

    def __setstate__(self, state):
        pass


_unpickle_set_safe = {
    ('dpareto.hypervolume_improvement.harness', 'HypervolumeImprovementHarness'),
    ('numpy', 'ndarray'),
    ('numpy', 'dtype'),
    ('numpy.core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', 'scalar'),
}


def my_find_class(modname, clsname):
    # print("DEBUG unpickling: %(modname)s . %(clsname)s" % locals())
    if (modname, clsname) in _unpickle_set_safe:
        # print('Importing...')
        __import__(modname, level=0)
        return getattr(sys.modules[modname], clsname)
    else:
        # print('Ignoring...')
        return Dummy


class SafeUnpickler(pickle.Unpickler):
        find_class = staticmethod(my_find_class)


def extract_bo_results(filename):
    print("Extracting results from pickled class...")
    obj = SafeUnpickler(open(filename, 'rb')).load()
    print("results extracted.")
    return obj._initial_privacy_vals, obj._initial_utility_vals, obj._all_results

def test():
    file_name = './mnist/dp_mnist_128_64_sgd/results/hvpoi_results/1546553409486/saved_optimization_state/round-256.pkl'
    file = open(file_name, 'rb')
    obj = SafeUnpickler(file).load()
    # print(obj.__dict__)
    print(obj._all_results)


if __name__ == '__main__':
    test()
