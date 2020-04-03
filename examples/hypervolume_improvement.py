# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from dpareto.models.adult.lr.dp_adam import DpAdultLrAdam
from dpareto.random_sampling.harness import RandomSamplingHarness
from dpareto.hypervolume_improvement.harness import HypervolumeImprovementHarness


def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Seed Bayeisan Optimization loop with initial points from a random sampling run
    num_initial_points = 32
    initial_data_options = {}
    initial_data_options['num_points'] = num_initial_points
    initial_data_options['filename'] = current_dir + '/random_sampling_results/1577508593819/full_results.pkl'

    # Set anti-ideal point for Pareto front optimization
    anti_ideal_point = [10, 0.999]

    # Hyperparameter range to search over
    optimization_domain = {}
    optimization_domain['epochs'] = [2, 4]
    optimization_domain['lot_size'] = [32, 128]
    optimization_domain['lr'] = [1e-3, 1e-2]
    optimization_domain['l2_clipping_bound'] = [0.1, 2]
    optimization_domain['z'] = [1, 8]
    # Number of new points to acquire via Bayesian optimization search
    num_instances = 2

    # Fixed hyperparameters
    fixed_hyperparams = {'fixed_delta': 1e-5, 'beta_1': 0.9, 'beta_2': 0.999}

    # Misc options for training procedure
    instance_options = {'use_gpu': False, 'verbose': True, 'compute_epoch_accuracy': False}

    # Save results in same directory
    output_dir = current_dir

    # Number of times each model training procedure will be repeated for each hyperparameter setting
    num_replications = 1
    
    # Number of parallel instatiations to use
    num_workers = 8

    # Pareto front plotting options
    plot_options = {'bottom': 1e-1, 'top': .34, 'left': 1e-2, 'right': 10}

    # Instantiate harness and run
    harness = HypervolumeImprovementHarness(DpAdultLrAdam, 'dp_adult_lr_adam', initial_data_options, anti_ideal_point, optimization_domain, fixed_hyperparams, instance_options, num_instances, num_replications, num_workers, plot_options, output_dir)
    harness.run()


if __name__ == "__main__":
    main()
