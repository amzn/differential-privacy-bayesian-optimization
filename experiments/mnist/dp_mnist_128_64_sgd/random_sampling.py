# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from dpareto.models.mnist.dp_sgd import DpMnistSgd
from dpareto.random_sampling.harness import RandomSamplingHarness


# Small run for testing purposes.
def main():
    # hyperparameter ranges to sample uniformly from
    hyperparam_distributions = {}
    hyperparam_distributions['epochs'] = {'type': 'uniform', 'params': [1, 400],
                                          'round_to_int': True}
    hyperparam_distributions['lot_size'] = {'type': 'normal', 'params': [800, 800],
                                            'round_to_int': True, 'reject_less_than': 16, 'reject_greater_than': 4000}
    hyperparam_distributions['lr'] = {'type': 'exponential', 'params': [10, 0.001],
                                      'reject_greater_than': 0.5}
    hyperparam_distributions['l2_clipping_bound'] = {'type': 'exponential', 'params': [0.5, 0.1],
                                                     'reject_greater_than': 12.0}
    hyperparam_distributions['z'] = {'type': 'exponential', 'params': [0.5, 0.1],
                                     'reject_greater_than': 16.0}
    hyperparam_distributions['fixed_delta'] = {'type': 'deterministic', 'value': 1e-5}

    instance_options = {'use_gpu': True, 'verbose': False, 'accumulate_privacy': True, 'hidden_layers': [128, 64]}

    output_dir = os.path.dirname(os.path.realpath(__file__)) + '/results'

    num_instances = 272
    num_replications = 10
    num_workers = 4
    harness = RandomSamplingHarness(DpMnistSgd, 'dp_mnist_128_64_sgd', hyperparam_distributions, instance_options, num_instances, num_replications, num_workers, output_dir)
    harness.run()


if __name__ == "__main__":
    main()
