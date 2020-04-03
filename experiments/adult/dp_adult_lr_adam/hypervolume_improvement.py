# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from dpareto.models.adult.lr.dp_adam import DpAdultLrAdam
from dpareto.random_sampling.harness import RandomSamplingHarness
from dpareto.hypervolume_improvement.harness import HypervolumeImprovementHarness


def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    num_initial_points = 16
    initial_data_options = {}
    initial_data_options['num_points'] = num_initial_points

    anti_ideal_point = [10, 0.999]

    optimization_domain = {}
    optimization_domain['epochs'] = [1, 64]
    optimization_domain['lot_size'] = [8, 512]
    optimization_domain['lr'] = [5e-4, 5e-2]
    optimization_domain['l2_clipping_bound'] = [.1, 4]
    optimization_domain['z'] = [.1, 16]

    fixed_hyperparams = {'fixed_delta': 1e-5, 'beta_1': 0.9, 'beta_2': 0.999}

    instance_options = {'use_gpu': False, 'verbose': True, 'compute_epoch_accuracy': False}

    plot_options = {'bottom': 1e-1, 'top': .34, 'left': 1e-2, 'right': 10}

    output_dir = current_dir + '/results'
    
    num_instances = 256
    num_replications = 10
    num_workers = 8
    harness = HypervolumeImprovementHarness(DpAdultLrAdam, 'dp_adult_lr_adam', initial_data_options, anti_ideal_point, optimization_domain, fixed_hyperparams, instance_options, num_instances, num_replications, num_workers, plot_options, output_dir)
    harness.run()


if __name__ == "__main__":
    main()
