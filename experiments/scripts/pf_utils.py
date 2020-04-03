# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import numpy as np
import pandas as pd

import experiments.scripts.picky_unpickler as pu


def get_all_bo_points(filename):
    initial_privacy_vals, initial_utility_vals, non_aggregated_points = pu.extract_bo_results(filename)

    initial_privacy_vals = [x[0] for x in initial_privacy_vals]
    initial_utility_vals = [x[0] for x in initial_utility_vals]
    initial_points = list(zip(initial_privacy_vals, initial_utility_vals))
    
    aggregated_points = aggregate_bo_points(non_aggregated_points)
    aggregated_points = initial_points + aggregated_points

    return aggregated_points

def get_all_rs_points(filename):
    full_results = pickle.load(open(filename,'rb'))
    aggregated_points = aggregate_rs_points(full_results)
    return aggregated_points

# Gets list of PF points sorted by increasing privacy
def compute_pf_points(points):
    sorted_points = sorted(points, key=lambda tup: tup[0])
    pf_points = []
    for i in range(len(sorted_points)):
        current_err = sorted_points[i][1]
        better_point_exists = False
        # check all points with better privacy, see if any have better error
        for j in range(i-1, -1, -1):
            if sorted_points[j][1] < current_err:
                better_point_exists = True
                break
        if not better_point_exists:
            pf_points.append(sorted_points[i])
    return pf_points

# Gets the volume of a given PF
def compute_pf_volume(pf_points, anti_ideal_point, log_transform_x=False):
    # remove all points that have privacy worse than anti-ideal privacy
    pf_points = [x for x in pf_points if x[0] < anti_ideal_point[0]]
    # remove all points that have privacy worse than anti-ideal utility
    pf_points = [x for x in pf_points if x[1] < anti_ideal_point[1]]
    
    volume = 0
    # compute volume from left to right (excluding final point)
    for i in range(len(pf_points)-1):
        if log_transform_x:
            width = np.log(pf_points[i+1][0]) - np.log(pf_points[i][0])
        else:
            width = pf_points[i+1][0] - pf_points[i][0]
        height = anti_ideal_point[1] - pf_points[i][1]
        volume += width * height
    
    # compute volume for final (right-most) point
    if log_transform_x:
        width = np.log(anti_ideal_point[0]) - np.log(pf_points[-1][0])
    else:
        width = anti_ideal_point[0] - pf_points[-1][0]
    height = anti_ideal_point[1] - pf_points[-1][1]
    volume += width * height
    
    return volume

# Aggregates the BO results by computing the mean of the errors for replications
def aggregate_bo_points(non_aggregated_points):
    aggregated_points = []
    i = 0
    while i < len(non_aggregated_points):
        privacy_val = non_aggregated_points[i][0]
        utility_val = non_aggregated_points[i][1]
        j = i + 1
        while j < len(non_aggregated_points) and non_aggregated_points[j][0] == -2:
            utility_val += non_aggregated_points[j][1]
            j += 1
        utility_val /= (j - i)
        aggregated_points.append((privacy_val, 1-utility_val))
        i = j
    return aggregated_points

# Aggregates the RS results by computing the mean of the errors for replications
def aggregate_rs_points(full_results):
    aggregated_points = {}
    for point in full_results:
        hyperparams_dict = point[0]
        frozen_hyperparams = frozenset(hyperparams_dict.items())
        if frozen_hyperparams not in aggregated_points:
            aggregated_points[frozen_hyperparams] = ([], [])
        aggregated_points[frozen_hyperparams][0].append(point[1])
        aggregated_points[frozen_hyperparams][1].append(point[2])
    
    processed_data = []
    seen_points = set()
    for point in full_results:
        hyperparams_dict = point[0]
        frozen_hyperparams = frozenset(hyperparams_dict.items())
        if frozen_hyperparams not in seen_points:
            seen_points.add(frozen_hyperparams)
            # compute privacy as max value
            privacy = np.max(aggregated_points[frozen_hyperparams][0])
            # compute utility as mean value
            utility = np.mean(aggregated_points[frozen_hyperparams][1])
            # add everything back into the list of processed data
            processed_data.append((hyperparams_dict, privacy, utility))
    
    aggregated_points = [(point[1], 1-point[2]) for point in processed_data]
    return aggregated_points

def make_plottable_dataframe(pf_points, max_x=20, max_y=20):
    # Add helper points in
    plottable_points = [(pf_points[0][0], max_y)]
    for i in range(len(pf_points)-1):
        plottable_points.append(pf_points[i])
        plottable_points.append((pf_points[i+1][0], pf_points[i][1]))
    plottable_points.append(pf_points[-1])
    plottable_points.append((max_x, pf_points[-1][1]))

    return pd.DataFrame({'epsilon': [x[0] for x in plottable_points], 'error': [x[1] for x in plottable_points]}) 

def test(rs_filename, bo_filename, anti_ideal_point, num_points):
    rs_points = get_all_rs_points(rs_filename)
    bo_points = get_all_bo_points(bo_filename)
    
    initial_rs_pf_points = compute_pf_points(rs_points[:16])
    initial_rs_volume = compute_pf_volume(initial_rs_pf_points, anti_ideal_point)
    print("Initial RS: {}".format(initial_rs_volume))
    
    initial_bo_pf_points = compute_pf_points(bo_points[:16])
    initial_bo_volume = compute_pf_volume(initial_bo_pf_points, anti_ideal_point)
    print("Initial BO: {}".format(initial_bo_volume))

    final_rs_pf_points = compute_pf_points(rs_points[:num_points])
    final_rs_volume = compute_pf_volume(final_rs_pf_points, anti_ideal_point)
    print("Final RS: {}".format(final_rs_volume))
    
    final_bo_pf_points = compute_pf_points(bo_points[:num_points])
    final_bo_volume = compute_pf_volume(final_bo_pf_points, anti_ideal_point)
    print("Final BO: {}".format(final_bo_volume))

if __name__ == '__main__':
    # Adult LR ADAM
    # rs_filename = 'experiments/adult/dp_adult_lr_adam/results/random_sampling_results/1543689523130/full_results.pkl'
    # bo_filename = 'experiments/adult/dp_adult_lr_adam/results/hvpoi_results/1547534280338/saved_optimization_state/round-256.pkl'
    # MNIST-128-64
    rs_filename = 'experiments/mnist/dp_mnist_128_64_sgd/results/random_sampling_results/1544097282583/full_results.pkl'
    bo_filename = 'experiments/mnist/dp_mnist_128_64_sgd/results/hvpoi_results/1546553409486/saved_optimization_state/round-256.pkl'
    
    anti_ideal_point = (10, 1)
    num_points = 272
    test(rs_filename, bo_filename, anti_ideal_point, num_points)
