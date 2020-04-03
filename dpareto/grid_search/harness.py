# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import multiprocessing as mp
from multiprocessing import Pool
import time
import os
import psutil
import numpy as np

from dpareto.utils.mxnet_scripts import get_gpu_count
from dpareto.utils.object_io import save_object
from dpareto.utils.random_seed_setter import set_random_seed


# This copies a good deal of code from RandomSamplingHarness
# Should probably be refactored later
class GridSearchHarness:
    def __init__(self, model_class, problem_name, hyperparam_ranges, instance_options={}, points_per_param=2, num_replications=1, num_workers=1, output_base_dir=''):
        mp.set_start_method('spawn', force=True)  # workaround for weird matplotlib+multiprocessing compatibility issue

        self._model_class = model_class
        self._problem_name = problem_name

        self._hyperparam_ranges = hyperparam_ranges
        self._instance_options = instance_options
        self._points_per_param = points_per_param
        self._num_replications = num_replications
        self._num_workers = num_workers

        # Overwrite hardcoded options with command-line-supplied options
        self._apply_args()

        # Multiprocessing worker info
        self._id_queue = mp.Manager().Queue()
        for pid in range(self._num_workers):
            self._id_queue.put(pid)

        # Output info
        self._output_base_dir = output_base_dir
        self._run_name = int(time.time() * 1000)
        self._make_results_directory()

    def _apply_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--device')
        parser.add_argument('--workers')
        parser.add_argument('--points_per_param')
        args = parser.parse_args()

        # Set device if provided
        if args.device is not None:
            if args.device == 'gpu':
                self._instance_options['use_gpu'] = True
            else:
                self._instance_options['use_gpu'] = False
        # Otherwise, default to CPU if nothing is provided in instance_options
        else:
            if 'use_gpu' not in self._instance_options:
                # if not specified or if 'cpu'
                self._instance_options['use_gpu'] = False

        # Set points per param if provided.
        if args.points_per_param is not None:
            self._points_per_param = int(args.points_per_param)

        # Set number of workers if provided. An explicit number or 'max' can be provided.
        if args.workers is not None:
            if args.workers == 'max':
                # get number of CPU cores if use_gpu is False
                if not self._instance_options['use_gpu']:
                    self._num_workers = mp.cpu_count()
                else:  # get number of GPU cores
                    self._num_workers = get_gpu_count()
                    if self._num_workers == 0:
                        print("No GPUs available. Exiting.")
                        exit()
            else:
                self._num_workers = int(args.workers)

    # Main runner of the grid search framework
    def run(self):
        self._generate_hyperparameter_mesh_instances()
        self._run_all_instances()
        self._store_results()
        return self._results

    # Generates grid mesh hyperparameter setting instances
    def _generate_hyperparameter_mesh_instances(self):
        print("Generating hyperparameters mesh...", end='')
        linspaces = []
        param_names = []
        for param_name, range_descr in self._hyperparam_ranges.items():
            if 'value' in range_descr:
                # fixed value, attach to each instance after generation
                continue

            ls = np.linspace(range_descr['min'], range_descr['max'], self._points_per_param)
            linspaces.append(ls)
            param_names.append(param_name)

        instances = np.array(np.meshgrid(*linspaces)).T.reshape(-1, len(linspaces))

        for param_name, range_descr in self._hyperparam_ranges.items():
            if 'value' in range_descr:
                column_to_add = np.full((instances.shape[0], 1), range_descr['value'])
                instances = np.hstack((instances, column_to_add))
                param_names.append(param_name)

        self._hyperparam_instances = []
        for instance in instances:
            hyperparam_instance = {}
            for i, param_name in enumerate(param_names):
                range_descr = self._hyperparam_ranges[param_name]
                if range_descr.get('round_to_int', False):
                    hyperparam_instance[param_name] = int(round(instance[i]))
                else:
                    hyperparam_instance[param_name] = instance[i]
            for j in range(self._num_replications):
                self._hyperparam_instances.append(hyperparam_instance)

        save_object(self._results_dir, self._hyperparam_instances, 'hyperparams')
        print("done. Generated and saved.")

    # From the instances (and their replications) of hyperparameters generated, coordinates the parallel runs
    # of all the training instances corresponding to each hyperparameter instance
    def _run_all_instances(self):
        print("Beginning {} runs distributed across {} workers.".format(len(self._hyperparam_instances), self._num_workers))

        # Create a list of inputs for the to-be-spawned process
        training_instances_inputs = []
        for idx, hyperparam_instance in enumerate(self._hyperparam_instances):
            single_instance_hyperparams = hyperparam_instance.copy()
            single_instance_options = self._instance_options.copy()
            # Compute privacy only for the first instance of its kind, not for any subsequent replications. Determine
            # this simply using the ordering of the instances in the hyperparam_instances list
            if idx > 0 and self._hyperparam_instances[idx] == self._hyperparam_instances[idx-1]:
                single_instance_options['accumulate_privacy'] = False
            training_instance_input = {'hyperparams': single_instance_hyperparams,
                                       'options': single_instance_options,
                                       'model_class': self._model_class}
            training_instances_inputs.append(training_instance_input)

        # Begin multiprocessing across the instances
        with Pool(self._num_workers, self._process_id_init, (self._id_queue,)) as pool:
            results = pool.map_async(self._run_single_instance, training_instances_inputs, chunksize=1)
            while not results.ready():
                completed_instances = len(self._hyperparam_instances) - results._number_left
                print("Status: {} of {} complete.".format(completed_instances, len(self._hyperparam_instances)), end='\r')
                time.sleep(1)
            self._results = results.get()

    # Given some hyperparameters and options, runs a single instance on its own compute device
    @staticmethod
    def _run_single_instance(inputs):
        hyperparams = inputs['hyperparams']
        options = inputs['options']
        model_class = inputs['model_class']

        global process_id
        options.update({'device_id': process_id})
        # Ensures the process is running on the CPU identified by process_id.
        proc = psutil.Process()
        try:
            proc.cpu_affinity([process_id])
        except:
            pass  # doesn't work on Macs...but Macs seem to handle CPU assignment fine without this, so just continue.

        # Change this process' random seed
        custom_seed = int(time.time() + hash(frozenset(hyperparams.items())) + 547*float(process_id)) # don't ask
        set_random_seed(custom_seed)

        print("Worker {} evaluating hyperparameters {}.".format(process_id, hyperparams))

        # Run instance with hyperparams
        priv, acc = -1, -1
        try:
            priv, acc = model_class(hyperparams, options).run()
            gc.collect()  # necessary to avoid out-of-memory errors
            if 'accumulate_privacy' in options and options['accumulate_privacy'] == False:
                priv = -2
        # If an instance crashes, just return (-1,-1) as a failure code
        except Exception as e:
            print("Worker {} running with hyperparameters \n{}\n threw the following exception:\n{}".format(process_id, hyperparams, e))
        finally:
            print("Worker {} returning result: ({}, {})".format(process_id, priv, acc))
            return priv, acc

    # Gets the unique process id (to be used to assign a unique compute device)
    @staticmethod
    def _process_id_init(queue):
        global process_id
        process_id = queue.get()

    def _store_results(self):
        full_results = [(self._hyperparam_instances[i], self._results[i][0], self._results[i][1])
                        for i in range(len(self._hyperparam_instances))]
        save_object(self._results_dir, full_results, 'full_results')

    def _make_results_directory(self):
        if not self._output_base_dir:
            self._results_dir = 'results/grid_search_results/{}/{}'.format(self._problem_name, self._run_name)
        else:
            self._results_dir = '{}/grid_search_results/{}'.format(self._output_base_dir, self._run_name)
        os.makedirs(self._results_dir, exist_ok=True)
        self._instance_options['output_base_dir'] = self._results_dir
