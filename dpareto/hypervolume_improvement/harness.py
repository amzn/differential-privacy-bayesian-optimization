# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import os
import pickle
import psutil
import time
import argparse

import gpflow
import gpflowopt
import numpy as np
import scipy as sp
import tensorflow as tf

from dpareto.utils.object_io import save_object
from dpareto.utils.random_seed_setter import set_random_seed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disables TensorFlow debugging info dump that appears on Windows machines
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # forces TensorFlow to use CPU instead of defaulting to GPU

class HypervolumeImprovementHarness:
    def __init__(self, model_class, problem_name, initial_data_options, anti_ideal_point, optimization_domain, fixed_hyperparams, instance_options={}, num_instances=0, num_replications=1, num_workers=1, plot_options={}, output_base_dir=''):
        mp.set_start_method('spawn', force=True)  # workaround for weird matplotlib+multiprocessing compatibility issue
        
        self._model_class = model_class
        self._problem_name = problem_name
        self._num_instances = num_instances
        self._num_replications = num_replications

        # This is used as an ordering for consistently converting the hyperparams dict of
        # hyperparam-name-string -> hyperparam-value that we use into a numpy array of hyperparam
        # values that GPFlowOpt uses.
        self._optimizable_hyperparams_names = list(optimization_domain.keys())

        parser = argparse.ArgumentParser()
        parser.add_argument('--init-data-file')
        args = parser.parse_args()

        # Loads initial data into numpy arrays from file or generates it.
        if args.init_data_file is not None:
            initial_data_options["filename"] = args.init_data_file
        initial_data = self._get_initial_points(initial_data_options)
        self._initial_hyperparams_vals = initial_data[0]
        self._initial_privacy_vals = initial_data[1]
        self._initial_utility_vals = initial_data[2]
        self._max_initial_utility_variance = initial_data[3]

        # Transforms the initial data into a format that GPFlowOpt can use.
        transformed_data = self._create_initial_transformations(initial_data)
        self._transformed_initial_hyperparams_vals = transformed_data[0]
        self._transformed_initial_privacy_vals = transformed_data[1]
        self._transformed_initial_utility_vals = transformed_data[2]

        # Creates separate GPs for modeling privacy and utility.
        self._privacy_gp = self._create_privacy_gp()
        self._utility_gp = self._create_utility_gp()

        # Runs a hard-coded sanity check against data from the loaded file.
        self._print_sanity_check()

        # Sets the information for the reference (aka anti-ideal) point and its transformation.
        self._anti_ideal_point = anti_ideal_point
        transformed_anti_ideal_eps = self._transform_privacy(self._anti_ideal_point[0])
        transformed_anti_ideal_err = self._transform_utility(self._anti_ideal_point[1])
        self._transformed_anti_ideal_point = [transformed_anti_ideal_eps, transformed_anti_ideal_err]

        # Optimization domain dictates the range of each to-be-optimized-over hyperparamter. Fixed hyperparams
        # are the hyperparams that aren't optimized over.
        self._optimization_domain = optimization_domain
        self._gpflowopt_domain = self._construct_gpflowopt_domain(optimization_domain)
        self._fixed_hyperparams = fixed_hyperparams
        self._instance_options = instance_options

        # Create acquisition function, set it's anti-ideal point, and create it's optimizer.
        self._acquisition = gpflowopt.acquisition.HVProbabilityOfImprovement([self._privacy_gp, self._utility_gp])
        self._acquisition.reference = np.array([self._transformed_anti_ideal_point])
        self._acquisition_optimizer = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(self._gpflowopt_domain, 5000),
                                                                       gpflowopt.optim.SciPyOptimizer(self._gpflowopt_domain)])

        # Create the Bayesian Optimizer.
        self._bayesian_optimizer = gpflowopt.BayesianOptimizer(self._gpflowopt_domain,
                                                               self._acquisition,
                                                               optimizer=self._acquisition_optimizer,
                                                               scaling=False)

        # Multiprocessing worker info
        self._num_workers = num_workers

        # Misc fields
        self._saved_hypervolumes = []
        self._all_results = []
        self._output_base_dir = output_base_dir
        self._run_name = int(time.time() * 1000)
        self._save_results = True
        self._make_results_directory()
        self._show_plot = plot_options.get('show_plot', False)
        self._plot_options = plot_options


    ##################################################################################
    # Initial data-related methods
    ##################################################################################
    @staticmethod
    def _process_replications(data):
        aggregated_points = {}
        for point in data:
            hyperparams_dict = point[0]
            frozen_hyperparams = frozenset(hyperparams_dict.items())
            if frozen_hyperparams not in aggregated_points:
                aggregated_points[frozen_hyperparams] = ([], [])
            aggregated_points[frozen_hyperparams][0].append(point[1])
            aggregated_points[frozen_hyperparams][1].append(point[2])
        
        processed_data = []
        seen_points = set()
        for point in data:
            hyperparams_dict = point[0]
            frozen_hyperparams = frozenset(hyperparams_dict.items())
            if frozen_hyperparams not in seen_points:
                seen_points.add(frozen_hyperparams)
                # compute privacy as max value
                privacy = np.max(aggregated_points[frozen_hyperparams][0])
                # compute utility as mean value
                utility = np.mean(aggregated_points[frozen_hyperparams][1])
                # add variance of utility values
                utility_variance = np.var(aggregated_points[frozen_hyperparams][1])
                # add everything back into the list of processed data
                processed_data.append((hyperparams_dict, privacy, utility, utility_variance))
        return processed_data

    def _get_initial_points(self, initial_data_options):
        num_initial_points = initial_data_options['num_points']

        # get initial data from file or generate it with random sampling
        if 'filename' in initial_data_options:
            filename = initial_data_options['filename']
            raw_data = self._load_points_from_file(filename)
        else:
            raise ValueError("Please provide path to the file full_results.pkl that contains initial data points from random sampling run.")

        # Process out the replications and extract the proper number of initial points
        processed_data = self._process_replications(raw_data)[:num_initial_points]

        # Set input hyperparams and output privacy & utility
        initial_hyperparams_dicts = np.array([instance[0] for instance in processed_data])

        initial_hyperparams_vals = np.array([np.array([instance[name] for name in self._optimizable_hyperparams_names]) for instance in initial_hyperparams_dicts])
        initial_privacy_vals = np.expand_dims(np.array([instance[1] for instance in processed_data]), axis=1)
        initial_utility_vals = 1-np.expand_dims(np.array([instance[2] for instance in processed_data]), axis=1)
        max_initial_utility_variance = np.max([instance[3] for instance in processed_data])
        return initial_hyperparams_vals, initial_privacy_vals, initial_utility_vals, max_initial_utility_variance

    def _load_points_from_file(self, filename):
        with open(filename, 'rb') as f:
            raw_data = pickle.load(f)
        return raw_data

    def _create_initial_transformations(self, initial_data):
        transformed_initial_hyperparams_vals = self._transform_initial_hyperparameters(initial_data[0])
        transformed_initial_privacy_vals = self._transform_initial_privacy(initial_data[1])
        transformed_initial_utility_vals = self._transform_utility(initial_data[2])
        return transformed_initial_hyperparams_vals, transformed_initial_privacy_vals, transformed_initial_utility_vals

    ##################################################################################
    # Hyperparameter-related methods
    ##################################################################################
    def _transform_initial_hyperparameters(self, initial_hyperparams_vals):
        self._hyperparam_normalization_factor = initial_hyperparams_vals.max(axis=0)
        return self._transform_hyperparameters(initial_hyperparams_vals)

    def _transform_hyperparameters(self, hyperparams):
        return hyperparams / self._hyperparam_normalization_factor

    def _untransform_hyperparameters(self, transformed_hyperparams):
        return transformed_hyperparams * self._hyperparam_normalization_factor

    ##################################################################################
    # Privacy-related methods
    ##################################################################################
    def _transform_initial_privacy(self, initial_privacy_vals):
        log_initial_privacy_vals = np.log(initial_privacy_vals)
        self._privacy_normalization_factor = np.asscalar(log_initial_privacy_vals.max(axis=0))
        return self._transform_privacy(initial_privacy_vals)

    def _transform_privacy(self, val):
        return np.log(val) / self._privacy_normalization_factor

    def _untransform_privacy(self, transformed_val):
        return np.exp(transformed_val * self._privacy_normalization_factor)

    def _create_privacy_gp(self):
        privacy_gp = gpflow.gpr.GPR(self._transformed_initial_hyperparams_vals.copy(),
                                    self._transformed_initial_privacy_vals.copy(),
                                    self.SafeMatern52(input_dim=5, ARD=True), name='privacy')
        privacy_gp.likelihood.variance = 0
        privacy_gp.likelihood.variance.fixed = True
        privacy_gp.optimize()
        return privacy_gp

    def predict_privacy(self, hyperparams, transform_hyperparams=False, untransform_prediction=True):
        gp_input = self._transform_hyperparameters(hyperparams) if transform_hyperparams else hyperparams
        pred = self._privacy_gp.predict_f(gp_input)[0]
        if untransform_prediction:
            pred = self._untransform_privacy(pred)
        return pred

    ##################################################################################
    # Utility-related methods
    ##################################################################################
    def _transform_initial_utility(self, initial_utility_vals):
        return self._transform_utility(initial_utility_vals)

    def _transform_utility(self, val):
        return sp.special.logit(val)

    def _untransform_utility(self, transformed_val):
        return sp.special.expit(transformed_val)

    def _create_utility_gp(self):
        utility_gp = gpflow.gpr.GPR(self._transformed_initial_hyperparams_vals.copy(),
                                    self._transformed_initial_utility_vals.copy(),
                                    self.SafeMatern52(input_dim=5, ARD=True), name='utility')
        utility_gp.likelihood.variance = 0.01
        utility_gp.optimize()
        return utility_gp

    def predict_utility(self, hyperparams, transform_hyperparams=False, untransform_prediction=True):
        gp_input = self._transform_hyperparameters(hyperparams) if transform_hyperparams else hyperparams
        pred = self._utility_gp.predict_f(gp_input)[0]
        if untransform_prediction:
            pred = self._untransform_utility(pred)
        return pred

    ##################################################################################
    # Multi-objective optimization-related methods
    ##################################################################################
    def predict(self, hyperparams, transform_hyperparams=False, untransform_prediction=True):
        privacy_pred = self.predict_privacy(hyperparams, transform_hyperparams, untransform_prediction)
        utility_pred = self.predict_utility(hyperparams, transform_hyperparams, untransform_prediction)
        return privacy_pred, utility_pred

    def _construct_gpflowopt_domain(self, optimization_domain):
        gpflowopt_domain = None
        for idx, name in enumerate(self._optimizable_hyperparams_names):
            transformed_domain = optimization_domain[name] / self._hyperparam_normalization_factor[idx]
            new_dim = gpflowopt.domain.ContinuousParameter(name, *transformed_domain)
            if not gpflowopt_domain:
                gpflowopt_domain = new_dim
            else:
                gpflowopt_domain += new_dim
        return gpflowopt_domain

    def get_transformed_hypervolume(self):
        hv = self._acquisition.pareto.hypervolume(self._transformed_anti_ideal_point)
        return hv

    def get_untransformed_hypervolume(self):
        # Get untransformed pf points
        pf, dom = gpflowopt.pareto.non_dominated_sort(self._acquisition.data[1])
        pf_eps = self._untransform_privacy(pf[:, 0])
        pf_err = self._untransform_utility(pf[:, 1])
        # Sort both lists by increasing x-axis (eps)
        pf_eps, pf_err = (list(t) for t in zip(*sorted(zip(pf_eps, pf_err))))

        # TODO: Do this before untransforming (in case the untransform doesn't preserve ordering).
        # Beginning with the right-most point on the pf, discard out of both lists if it is not in the region-of-interest
        for i in range(len(pf_eps)-1, -1):
            if pf_eps > self._anti_ideal_point[0] or pf_err > self._anti_ideal_point[1]:
                del pf_eps[i]
                del pf_err[i]

        hv = 0
        if len(pf_eps) > 0:
            for i in range(len(pf_eps)-1, 0):
                # Accumulate the rectangle formed by: 1) current point to the anti-ideal privacy point, 2) current point up to the next point.
                hv += ((self._anti_ideal_point[0] - pf_eps[i]) * (pf_err[i-1] - pf_err[i]))
            # Final point on the pf has rectangle formed by: 1) current point to the anti-ideal privacy point, 2) current point to the anti-ideal error point.
            hv += ((self._anti_ideal_point[0] - pf_eps[0]) * (self._anti_ideal_point[1] - pf_err[0]))
        return hv

    def run(self):
        self._previous_hyperparams = None
        self._optimization_point_number = 0

        optimization_result = self._bayesian_optimizer.optimize([self._optimization_handler], n_iter=self._num_instances)

        # Post final run tasks
        hypervolume = self.get_untransformed_hypervolume()
        self._saved_hypervolumes.append(hypervolume)
        print("Hypervolume is: {}".format(hypervolume))
        self._estimate_last_point()
        self._plot_and_save()

    ##################################################################################
    # Methods relating to the expensive function being optimized over.
    ##################################################################################
    def _convert_model_input_to_problem_instance_input(self, transformed_hyperparams_arr):
        hyperparams_arr = self._untransform_hyperparameters(transformed_hyperparams_arr)
        hyperparams_dict = self._fixed_hyperparams.copy()
        # TODO: make better
        for idx, name in enumerate(self._optimizable_hyperparams_names):
            val = hyperparams_arr[0][idx]
            if name in ['epochs', 'lot_size']:
                val = int(round(val))
            hyperparams_dict[name] = val
        return hyperparams_dict

    def _run_pre_iteration_tasks(self, transformed_hyperparams):
        # Print and store hypervolume from previous run
        hypervolume = self.get_untransformed_hypervolume()
        self._saved_hypervolumes.append(hypervolume)
        print("Hypervolume is: {}".format(hypervolume))

        # Run check to ensure previously acquired point was learned properly, then save current hyperparams for next run.
        self._estimate_last_point()
        self._previous_hyperparams = transformed_hyperparams

        # Untransform hyperparams and put into our expected format (a dictionary of hyperparam names to vals).
        hyperparams_dict = self._convert_model_input_to_problem_instance_input(transformed_hyperparams)

        # Predict privacy and utility before training model.
        estimated_privacy, estimated_utility = self.predict(transformed_hyperparams, transform_hyperparams=False)
        print("curr point: {}".format(hyperparams_dict))
        print("curr pts estimated (eps, err): {}".format((estimated_privacy, estimated_utility)))

        self._plot_and_save()

        self._optimization_point_number += 1

        save_object('{}/optimization-hyperparameters'.format(self._results_dir),
                    hyperparams_dict,
                    'round-{}'.format(self._optimization_point_number))
        
        return hyperparams_dict

    def _run_post_iteration_tasks(self, eps, acc):
        self._print_sanity_check()

        transformed_eps = self._transform_privacy(eps)
        transformed_err = self._transform_utility(1-acc)
        
        return transformed_eps, transformed_err

    def _run_replications(self, hyperparams_dict, options):
        print("Beginning run of {} replications using {} workers.".format(self._num_replications, self._num_workers))

        # Create a list of inputs for the to-be-spawned process
        training_instances_inputs = []
        for i in range(self._num_replications):
            process_hyperparams = hyperparams_dict.copy()
            process_options = options.copy()
            # Compute privacy only for the first instance of its kind, not for any subsequent replications. Determine
            # this simply using the ordering of the instances in the hyperparam_instances list
            if i > 0:
                process_options['accumulate_privacy'] = False
                process_options['verbose'] = False
            training_instance_input = {'hyperparams': process_hyperparams,
                                       'options': process_options,
                                       'model_class': self._model_class}
            training_instances_inputs.append(training_instance_input)

        # Begin multiprocessing across the instances
        self._id_queue = mp.Manager().Queue()
        for pid in range(self._num_workers):
            self._id_queue.put(pid)
        with Pool(self._num_workers, self._process_id_init, (self._id_queue,)) as pool:
            results = pool.map_async(self._run_single_instance, training_instances_inputs, chunksize=1)
            # while not results.ready():
            #     completed_instances = self._num_replications - results._number_left
            #     print("Status: {} of {} complete.".format(completed_instances, self._num_replications), end='\r')
            #     time.sleep(1)
            self._all_results += results.get()

        # process results
        aggregated_privacy = np.max([result[0] for result in results.get()])
        aggregated_utility = np.mean([result[1] for result in results.get()])
        return aggregated_privacy, aggregated_utility

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

        # Run instance with random hyperparams
        priv, acc = -1, -1
        try:
            instance = model_class(hyperparams, options)
            priv, acc = instance.run()
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

    def _optimization_handler(self, transformed_hyperparams):
        # GPFlowOpt discards output on Windows; the following lines correct this.
        import sys
        sys.stdout = sys.__stdout__

        # Do any computation before testing a point on the objective function.
        hyperparams_dict = self._run_pre_iteration_tasks(transformed_hyperparams)

        options = self._instance_options
        options['output_base_dir'] = self._results_dir

        eps, acc = self._run_replications(hyperparams_dict, options)
        print("actual (eps, err): {}".format((eps, 1-acc)))

        # Do any computation after testing a point on the objective function before updating the model.
        transformed_eps, transformed_err = self._run_post_iteration_tasks(eps, acc)

        return np.array([(transformed_eps, transformed_err)])

    def _estimate_last_point(self):
        if self._previous_hyperparams is not None:
            estimated_privacy, estimated_utility = self.predict(self._previous_hyperparams, transform_hyperparams=False)
            print("#######################################################################################")
            print("Checking last point acquired in optimization loop.")
            print("estimated (eps, err) of point: {}".format((estimated_privacy, estimated_utility)))
            print("#######################################################################################")

    ##################################################################################
    # Miscellaneous code.
    ##################################################################################
    def _plot_and_save(self):
        self._plot_empirical_pareto_frontier(suffix=str(self._optimization_point_number))

        # save_object('{}/saved_gp_models'.format(self._results_dir),
        #             [self._privacy_gp, self._utility_gp],
        #             'round-{}'.format(self._optimization_point_number),
        #             save_text=False)

        save_object('{}/saved_optimization_state'.format(self._results_dir),
                    self,
                    'round-{}'.format(self._optimization_point_number),
                    save_text=False)

    def _plot_empirical_pareto_frontier(self, suffix=''):
        pf, dom = gpflowopt.pareto.non_dominated_sort(self._acquisition.data[1])
        pf_eps = self._untransform_privacy(pf[:, 0])
        pf_err = self._untransform_utility(pf[:, 1])
        # Sort both lists by increasing x-axis (eps)
        pf_eps, pf_err = (list(t) for t in zip(*sorted(zip(pf_eps, pf_err))))

        all_eps = self._untransform_privacy(self._acquisition.data[1][:, 0])
        all_err = self._untransform_utility(self._acquisition.data[1][:, 1])

        plt.figure(figsize=(9, 7))

        # Plot all points, color by dominance
        plt.scatter(all_eps, all_err, c=dom, marker='.')

        # Mark all points on the pf with red X and build a stepwise dominance line
        plt.scatter(pf_eps, pf_err, color='r', marker='x')
        boundary_pt = ((pf_eps[0], 2*self._anti_ideal_point[1]), (2*self._anti_ideal_point[0], pf_eps[-1]))
        pf_eps.insert(0, boundary_pt[0][0])
        pf_err.insert(0, boundary_pt[0][1])
        pf_eps.insert(len(pf_eps), boundary_pt[1][0])
        pf_err.insert(len(pf_err), boundary_pt[1][1])
        plt.step(pf_eps, pf_err, 'k-', where='post')

        # Plot options
        self._set_plot_properties(plt, self._plot_options)
        if self._save_results:
            plots_dir = '{}/pareto-front-plots'.format(self._results_dir)
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig('{}/round-{}.png'.format(plots_dir, suffix))
        if self._show_plot:
            # TODO: resolve blocking issue when showing plot on Windows
            plt.show()

    def _set_plot_properties(self, plt, plot_options):
        title = plot_options.get('title', 'Empirical Pareto Frontier')
        xlabel = plot_options.get('xlabel', 'epsilon')
        ylabel = plot_options.get('ylabel', 'classification error')
        plt.title(title, size=20)
        plt.xlabel(xlabel, size=18)
        plt.ylabel(ylabel, size=18)
        
        left = plot_options.get('left', 0)
        right = plot_options.get('right', self._anti_ideal_point[0])
        bottom = plot_options.get('bottom', 0)
        top = plot_options.get('top', self._anti_ideal_point[1])
        plt.xlim(left=left, right=right)
        plt.ylim(bottom=bottom, top=top)

    def _print_sanity_check(self):
        print("#######################################################################################")
        print("Running GP sanity check on a training point.")

        # first point in loaded data
        first_point = np.expand_dims(self._initial_hyperparams_vals[0], axis=0)

        actual_privacy = self._initial_privacy_vals[0]
        actual_utility = self._initial_utility_vals[0]
        print("actual (eps, err): {}".format((actual_privacy, actual_utility)))

        # gp estimate of first point
        estimated_privacy, estimated_utility = self.predict(first_point, transform_hyperparams=True)
        print("estimated (eps, err): {}".format((estimated_privacy, estimated_utility)))

        # perturbed first point in loaded data (lot size increase by 1)
        almost_first_point = np.copy(first_point)
        almost_first_point[0][self._optimizable_hyperparams_names.index('lot_size')] += 1
        estimated_privacy, estimated_utility = self.predict(almost_first_point, transform_hyperparams=True)
        print("estimated (eps, err) of perturbed: {}".format((estimated_privacy, estimated_utility)))

        print("#######################################################################################")

    def _make_results_directory(self):
        if not self._output_base_dir:
            self._results_dir = 'results/hvpoi_results/{}/{}'.format(self._problem_name, self._run_name)
        else:
            self._results_dir = '{}/hvpoi_results/{}'.format(self._output_base_dir, self._run_name)
        os.makedirs(self._results_dir, exist_ok=True)

    # Matern52 kernel isn't numerically safe, so this work-around is necessary.
    class SafeMatern52(gpflow.kernels.Matern52):
        def euclid_dist(self, X, X2):
            r2 = self.square_dist(X, X2)
            return tf.sqrt(r2 + 1e-7)
