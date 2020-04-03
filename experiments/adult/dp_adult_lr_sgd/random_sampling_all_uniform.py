# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
In this experiment all parameters are drawn from unifrom distributions
"""

import os

from dpareto.models.adult.lr.dp_sgd import DpAdultLrSgd
from dpareto.random_sampling.harness import RandomSamplingHarness

def main():
    # hyperparameter ranges to sample uniformly from
    hyperparam_distributions = {}
    hyperparam_distributions['epochs'] = {'type': 'uniform', 'params': [1, 64], 'round_to_int': True}
    hyperparam_distributions['lot_size'] = {'type': 'uniform', 'params': [8, 512], 'round_to_int': True}
    hyperparam_distributions['lr'] = {'type': 'uniform', 'params': [1e-3, 0.1]}
    hyperparam_distributions['l2_clipping_bound'] = {'type': 'uniform', 'params': [0.1, 4.0]}
    hyperparam_distributions['z'] = {'type': 'uniform', 'params': [0.1, 16.0]}
    hyperparam_distributions['fixed_delta'] = {'type': 'deterministic', 'value': 1e-5}

    instance_options = {'use_gpu': False, 'verbose': False, 'accumulate_privacy': True}

    output_dir = os.path.dirname(os.path.realpath(__file__)) + '/results'

    num_instances = 256
    num_replications = 10
    num_workers = 8
    harness = RandomSamplingHarness(DpAdultLrSgd, "dp_adult_lr_sgd", hyperparam_distributions, instance_options, num_instances, num_replications, num_workers, output_dir)
    harness.run()


if __name__ == "__main__":
    main()
