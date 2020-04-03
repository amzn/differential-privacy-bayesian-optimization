# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle

import autodp as adp
import dpareto.data.adult.importer as importer
from dpareto.utils.object_io import save_object


# TODO sample_fraction is generally not available as input to the algorithm!
def compute_epsilon(hyperparams, num_training_examples=None):

    if 'sample_fraction' in hyperparams:
        sample_fraction = hyperparams['sample_fraction']
        num_iterations = hyperparams['epochs'] * round(hyperparams['lot_size'] / sample_fraction)
    elif num_training_examples is not None:
        num_iterations = hyperparams['epochs'] * round(num_training_examples / hyperparams['lot_size'])
        sample_fraction = hyperparams['lot_size'] / num_training_examples
    else:
        raise ValueError('[compute_epsilon] Missing both num_training_examples and sample_fraction')

    accountant = adp.rdp_acct.anaRDPacct()
    cgf_func = lambda x: adp.rdp_bank.RDP_gaussian({'sigma': hyperparams['z']}, x)
    for i in range(num_iterations):
        accountant.compose_subsampled_mechanism(cgf_func, sample_fraction)
    return accountant.get_eps(hyperparams['fixed_delta'])


def get_adult_training_size():
    Xtrain, Ytrain, Xtest, Ytest = importer.import_adult_dataset(None)
    return len(Ytrain)
    # return 30956


def get_mnist_training_size():
    return 60000


def convert_from_pydiffpriv_to_autodp(num_training_examples, results_directory, results_name='full_results'):
    # get results
    original_results_base = '{}/{}'.format(results_directory, results_name)
    with open('{}.pkl'.format(original_results_base), 'rb') as f:
        original_results = pickle.load(f)

    # for each result, if privacy was computed, recompute privacy with autodp
    auto_dp_results = []
    for (hyperparams, pydiffpriv_privacy, utility) in original_results:
        if pydiffpriv_privacy > 0:
            autodp_privacy = compute_epsilon(hyperparams, num_training_examples)
        else:
            autodp_privacy = pydiffpriv_privacy
        # store result in same format as before
        auto_dp_results.append((hyperparams, autodp_privacy, utility))

    # rename old results to results_name+'-pydiffpriv' if it doesn't already exist
    if not os.path.isfile('{}-pydiffpriv.pkl'.format(original_results_base)):
        os.rename('{}.pkl'.format(original_results_base), '{}-pydiffpriv.pkl'.format(original_results_base))
        os.rename('{}.txt'.format(original_results_base), '{}-pydiffpriv.txt'.format(original_results_base))
    else:
        print('Old pydiffpriv files already exist. Continuing anyways.')

    # save new results to results_name
    save_object(results_directory, auto_dp_results, results_name)


if __name__ == "__main__":
    # Debug a problematic instance for autoDP
    Ntrain = get_adult_training_size()
    hp = {'fixed_delta': 1e-05, 'epochs': 2, 'lot_size': 8, 'lr': 0.006699102447124356, 'l2_clipping_bound': 2.909185033175768, 'z': 12.67021387220363}
    eps = compute_epsilon(hp, Ntrain)
    print(eps)
