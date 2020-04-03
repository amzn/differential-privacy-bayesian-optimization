# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from dpareto.models.adult.lr.dp_sgd import DpAdultLrSgd
from dpareto.grid_search.harness import GridSearchHarness


def main():
    # hyperparameter ranges to cover with grid mesh
    hyperparam_ranges = {}
    hyperparam_ranges['epochs'] = {'min': 1, 'max': 64, 'round_to_int': True}
    hyperparam_ranges['lot_size'] = {'min': 8, 'max': 512, 'round_to_int': True}
    hyperparam_ranges['lr'] = {'min': 1e-3, 'max': 0.1}
    hyperparam_ranges['l2_clipping_bound'] = {'min': 0.1, 'max': 4.0}
    hyperparam_ranges['z'] = {'min': 0.1, 'max': 16.0}
    hyperparam_ranges['fixed_delta'] = {'value': 1e-5}

    instance_options = {'use_gpu': False, 'verbose': False, 'accumulate_privacy': True}

    output_dir = os.path.dirname(os.path.realpath(__file__)) + '/results'

    points_per_param = 3

    num_replications = 10
    num_workers = 8
    harness = GridSearchHarness(DpAdultLrSgd, "dp_adult_lr_sgd", hyperparam_ranges, instance_options, points_per_param, num_replications, num_workers, output_dir)
    harness.run()


if __name__ == "__main__":
    main()
