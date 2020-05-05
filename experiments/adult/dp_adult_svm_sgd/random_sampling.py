# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from dpareto.models.adult.svm.dp_sgd import DpAdultSvmSgd
from dpareto.random_sampling.harness import RandomSamplingHarness


def main():
    # hyperparameter ranges to sample uniformly from
    hyperparam_distributions = {}
    hyperparam_distributions['epochs'] = {'type': 'uniform', 'params': [1, 64],
                                          'round_to_int': True}
    hyperparam_distributions['lot_size'] = {'type': 'normal', 'params': [128, 64],
                                            'round_to_int': True, 'reject_less_than': 8, 'reject_greater_than': 512}
    hyperparam_distributions['lr'] = {'type': 'exponential', 'params': [10, 1e-3],
                                      'reject_greater_than': 0.1}
    hyperparam_distributions['l2_clipping_bound'] = {'type': 'exponential', 'params': [0.1, 0.1],
                                                     'reject_greater_than': 4.0}
    hyperparam_distributions['z'] = {'type': 'exponential', 'params': [0.1, 0.1],
                                     'reject_greater_than': 16.0}
    hyperparam_distributions['fixed_delta'] = {'type': 'deterministic', 'value': 1e-5}

    instance_options = {'use_gpu': False, 'verbose': False, 'accumulate_privacy': True}

    output_dir = os.path.dirname(os.path.realpath(__file__)) + '/results'

    num_instances = 272
    num_replications = 10
    num_workers = 8
    harness = RandomSamplingHarness(DpAdultSvmSgd, "dp_adult_svm_sgd", hyperparam_distributions, instance_options,
                                    num_instances, num_replications, num_workers, output_dir)
    harness.run()


if __name__ == "__main__":
    main()
