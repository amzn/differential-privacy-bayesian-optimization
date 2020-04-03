# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from dpareto.models.adult.lr.dp_adam import DpAdultLrAdam
from dpareto.grid_search.harness import GridSearchHarness


def main():
    # Hyperparameter ranges to cover with grid mesh
    points_per_param = 2
    hyperparam_ranges = {}
    hyperparam_ranges['epochs'] = {'min': 2, 'max': 4, 'round_to_int': True}
    hyperparam_ranges['lot_size'] = {'min': 32, 'max': 128, 'round_to_int': True}
    hyperparam_ranges['lr'] = {'min': 1e-3, 'max': 1e-2}
    hyperparam_ranges['l2_clipping_bound'] = {'min': 0.1, 'max': 2.0}
    hyperparam_ranges['z'] = {'min': 1, 'max': 8}
    
    # Fixed hyperparameters
    hyperparam_ranges['fixed_delta'] = {'value': 1e-5}
    hyperparam_ranges['beta_1'] = {'value': 0.9}
    hyperparam_ranges['beta_2'] = {'value': 0.999}

    # Misc options for training procedure
    instance_options = {'use_gpu': False, 'verbose': False, 'accumulate_privacy': True}

    # Save results in same directory
    output_dir = os.path.dirname(os.path.realpath(__file__))

    # Number of times each model training procedure will be repeated for each hyperparameter setting
    num_replications = 1

    # Number of parallel instatiations to use
    num_workers = 8

    # Instantiate harness and run
    harness = GridSearchHarness(DpAdultLrAdam, "dp_adult_lr_adam", hyperparam_ranges, instance_options, points_per_param, num_replications, num_workers, output_dir)
    harness.run()


if __name__ == "__main__":
    main()
