# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from dpareto.models.adult.lr.dp_adam import DpAdultLrAdam
from dpareto.random_sampling.harness import RandomSamplingHarness

def main():
    # Sampling distributions for each individual hyperparameter
    hyperparam_distributions = {}
    hyperparam_distributions['epochs'] = {'type': 'uniform', 'params': [2, 4],
                                          'round_to_int': True}
    hyperparam_distributions['lot_size'] = {'type': 'normal', 'params': [64, 16],
                                            'round_to_int': True, 'reject_less_than': 32, 'reject_greater_than': 128}
    hyperparam_distributions['lr'] = {'type': 'exponential', 'params': [10, 1e-3],
                                      'reject_greater_than': 1e-2}
    hyperparam_distributions['l2_clipping_bound'] = {'type': 'exponential', 'params': [0.1, 0.1],
                                                     'reject_greater_than': 2.0}
    hyperparam_distributions['z'] = {'type': 'exponential', 'params': [0.1, 1],
                                     'reject_greater_than': 8.0}
    # Number of random hyperparameter settings to sample
    num_instances = 32
    
    # Fixed hyperparameters
    hyperparam_distributions['fixed_delta'] = {'type': 'deterministic', 'value': 1e-5}
    hyperparam_distributions['beta_1'] = {'type': 'deterministic', 'value': 0.9}
    hyperparam_distributions['beta_2'] = {'type': 'deterministic', 'value': 0.999}

    # Misc options for training procedure
    instance_options = {'use_gpu': False, 'verbose': False, 'accumulate_privacy': True}

    # Save results in same directory
    output_dir = os.path.dirname(os.path.realpath(__file__))

    # Number of times each model training procedure will be repeated for each hyperparameter setting
    num_replications = 1

    # Number of parallel instatiations to use
    num_workers = 8

    # Instantiate harness and run
    harness = RandomSamplingHarness(DpAdultLrAdam, "dp_adult_lr_adam", hyperparam_distributions, instance_options, num_instances, num_replications, num_workers, output_dir)
    harness.run()


if __name__ == "__main__":
    main()
