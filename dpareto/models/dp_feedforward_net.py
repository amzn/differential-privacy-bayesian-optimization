# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mxnet as mx
from mxnet import nd
import numpy as np
# from pydiffpriv import dpacct
from autodp import rdp_acct
import time

from dpareto.utils.object_io import save_object
from dpareto.utils.progress_bar import printProgressBar


class DpFeedforwardNet:
    def __init__(self, hyperparams, options={}):
        self._hyperparams = hyperparams
        self._epochs = hyperparams['epochs']  # total number of 'virtual epochs' to run
        self._lot_size = hyperparams['lot_size']  # lot_size as in DP-SGD paper
        self._fixed_delta = hyperparams['fixed_delta']  # for (eps, delta)-DP, this is the fixed value for delta

        # misc options and their defaults (if unspecified)
        self._options = options
        self._optimizer_name = options.get('optimizer_name', 'dp_sgd')
        self._enable_mxnet_profiling = options.get('enable_mxnet_profiling', False)
        self._save_plots = options.get('save_plots', False)
        self._print_epoch_status = options.get('print_epoch_status', True)
        self._compute_epoch_accuracy = options.get('compute_epoch_accuracy', True)
        self._verbose = options.get('verbose', False)
        self._use_gpu = options.get('use_gpu', True)
        self._run_training = options.get('run_training', True)
        self._debugging = options.get('debugging', False)
        self._accumulate_privacy = options.get('accumulate_privacy', True)
        self._output_base_dir = options.get('output_base_dir', '')

        # TODO: switching between optimizers this way doesn't feel right
        if self._optimizer_name == 'dp_sgd':
            from dpareto.optimizers.dp_sgd import DpSgd as Optimizer
        elif self._optimizer_name == 'dp_adam':
            from dpareto.optimizers.dp_adam import DpAdam as Optimizer
        else:
            raise NotImplementedError("Optimizer not implemented.")
        self._optimizer = Optimizer

        # MXNet settings
        if self._use_gpu:
            ctx = mx.gpu(options['device_id']) if options.get('device_id', False) else mx.gpu()
        else:
            ctx = mx.cpu()
        self._data_ctx = ctx
        self._model_ctx = ctx

        self._name = 'dp_ffnn'

    def run(self):
        # Helper methods
        def get_random_lot(data_loader):
            return next(iter(data_loader))

        # Data importing, pre-processing, and loading
        num_training_examples, num_testing_examples, train_data_lot_iterator, train_data_eval_iterator, test_data = self._load_data()
        # parameters calculated from loaded data
        self._num_training_examples = num_training_examples
        self._num_testing_examples = num_testing_examples
        self._hyperparams['sample_fraction'] = self._lot_size / num_training_examples
        rounds_per_epoch = round(num_training_examples / self._lot_size)

        # Set up privacy accountant
        accountant = rdp_acct.anaRDPacct() # dpacct.anaCGFAcct()
        eps_sequence = []

        # Network structure creation
        self._create_network_params()

        # Loss function
        loss_func = self._get_loss_func()

        # Optimization procedure
        trainer = self._optimizer(self._hyperparams, self._net, self._params, loss_func, self._model_ctx, accountant)

        # begin profiling if enabled
        if self._enable_mxnet_profiling:
            from mxnet import profiler
            profiler.set_config(profile_all=True, aggregate_stats=True, filename='profile_output.json')
            profiler.set_state('run')

        # Training sequence
        rounds = round(self._epochs * rounds_per_epoch)
        loss_sequence = []
        current_epoch_loss = mx.nd.zeros(1, ctx=self._model_ctx)
        for t in range(1, rounds + 1):
            if self._verbose and self._print_epoch_status:
                # show current epoch progress
                epoch_number = 1 + (t - 1) // rounds_per_epoch
                epoch_progress = 1 + (t - 1) % rounds_per_epoch
                printProgressBar(epoch_progress, rounds_per_epoch, prefix='Epoch {} progress:'.format(epoch_number), length=50)

            if self._run_training:
                # prepare random lot of data for DPSGD step
                data, labels = get_random_lot(train_data_lot_iterator)
                data = data.as_in_context(self._model_ctx).reshape((-1, 1, self._input_layer))
                labels = labels.as_in_context(self._model_ctx)
            else:
                data, labels = [], []

            # perform DPSGD step
            lot_mean_loss = trainer.step(data, labels, accumulate_privacy=self._accumulate_privacy, run_training=self._run_training)

            loss_sequence.append(lot_mean_loss)
            current_epoch_loss += lot_mean_loss

            # no need to continue running training if NaNs are present
            if not np.isfinite(lot_mean_loss):
                self._run_training = False
                if self._verbose: print("NaN loss on round {}.".format(t))
            if self._params_not_finite():
                self._run_training = False
                if self._verbose: print("Non-finite parameters on round {}.".format(t))


            if self._accumulate_privacy and self._debugging:
                eps_sequence.append(accountant.get_eps(self._fixed_delta))

            # print some stats after an "epoch"
            if t % rounds_per_epoch == 0:
                if self._verbose:
                    print("Epoch {}  (round {})  complete.".format(t / rounds_per_epoch, t))
                    if self._run_training:
                        print("mean epoch loss: {}".format(current_epoch_loss.asscalar() * self._lot_size / self._num_training_examples))
                        if self._compute_epoch_accuracy:
                            print("training accuracy: {}".format(self._evaluate_accuracy(train_data_eval_iterator)))
                            print("testing accuracy: {}".format(self._evaluate_accuracy(test_data)))
                    if self._accumulate_privacy and self._debugging:
                        print("eps used: {}\n".format(eps_sequence[-1]))
                    print()
                current_epoch_loss = mx.nd.zeros(1, ctx=self._model_ctx)

        # end profiling if enabled
        if self._enable_mxnet_profiling:
            mx.nd.waitall()
            profiler.set_state('stop')
            print(profiler.dumps())

        # Make sure we don't report a bogus number
        if self._accumulate_privacy:
            final_eps = accountant.get_eps(self._fixed_delta)
        else:
            final_eps = -1

        test_accuracy = self._evaluate_accuracy(test_data)

        if self._save_plots or self._debugging:
            self._create_and_save_plots(t, eps_sequence, loss_sequence, final_eps, test_accuracy)

        return final_eps, test_accuracy

    def _create_network_params(self):
        # Set the base parameters for the network
        self._params = {}
        if len(self._hidden_layers) > 0:
            #  Allocate parameters for input layer -> first hidden layer
            self._params['W0'] = self._xavier_initialize_params(self._input_layer, self._hidden_layers[0], self._model_ctx)
            self._params['b0'] = self._xavier_initialize_params(1, self._hidden_layers[0], self._model_ctx)

            #  Allocate parameters for hidden layer i -> hidden layer i+1
            for i in range(1, len(self._hidden_layers)):
                str_index = str(i)
                self._params['W' + str_index] = self._xavier_initialize_params(self._hidden_layers[i-1], self._hidden_layers[i], self._model_ctx)
                self._params['b' + str_index] = self._xavier_initialize_params(1, self._hidden_layers[i], self._model_ctx)

            #  Allocate parameters for final hidden layer -> output layer
            str_index = str(len(self._hidden_layers))
            self._params['W' + str_index] = self._xavier_initialize_params(self._hidden_layers[-1], self._output_layer, self._model_ctx)
            self._params['b' + str_index] = self._xavier_initialize_params(1, self._output_layer, self._model_ctx)
        else:
            #  Allocate parameters for input layer -> output layer
            self._params['W0'] = self._xavier_initialize_params(self._input_layer, self._output_layer, self._model_ctx)
            self._params['b0'] = self._xavier_initialize_params(1, self._output_layer, self._model_ctx)

    @staticmethod
    def _xavier_initialize_params(num_in, num_out, context, magnitude=3):
        params_shape = (num_in, num_out)
        weight_scale = np.sqrt(magnitude * 2 / (num_in + num_out))
        return nd.random_normal(shape=params_shape, scale=weight_scale, ctx=context)

    # Main network function to compute forward pass of the data.
    def _net(self, in_data, params, training_mode=True):
        if training_mode:
            dot = nd.batch_dot
        else:
            dot = nd.dot

        # if there are no hidden layers, just compute the output as a linear combination
        if len(self._hidden_layers) == 0:
            return (dot(in_data, params['W0']) + params['b0']).reshape((in_data.shape[0], self._output_layer))

        #  Compute the first hidden layer
        h0_linear = dot(in_data, params['W0']) + params['b0']
        h0 = nd.relu(h0_linear)

        #  Compute the (i+1)^th hidden layer
        hprevious = h0
        for i in range(1, len(self._hidden_layers)):
            str_index = str(i)
            hcurrent_linear = dot(hprevious, params['W' + str_index]) + params['b' + str_index]
            hcurrent = nd.relu(hcurrent_linear)
            hprevious = hcurrent

        str_index = str(len(self._hidden_layers))
        yhat_linear = dot(hprevious, params['W' + str_index]) + params['b' + str_index]

        return yhat_linear.reshape((in_data.shape[0], self._output_layer))

    def _params_not_finite(self):
        for param in self._params.values():
            if not np.isfinite(param.asnumpy()).all():
                return True
        return False

    @staticmethod
    def _get_loss_func():
        raise NotImplementedError("Method not implemented.")

    def _load_data(self):
        raise NotImplementedError("Method not implemented.")

    def _evaluate_accuracy(self, data_iterator):
        raise NotImplementedError("Method not implemented.")

    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    # Plot and save results
    def _create_and_save_plots(self, rounds, eps_sequence, loss_sequence, final_eps, test_accuracy):
        # Estabilish output directory
        dir_name = np.abs(hash((time.time(), frozenset(self._hyperparams.items()))))
        if not self._output_base_dir:
            plots_subdir = self._name
            plots_dir = 'results/instance-results/{}/{}'.format(plots_subdir, dir_name)
        else:
            plots_dir = '{}/instance-results/{}'.format(self._output_base_dir, dir_name)
        import os
        os.makedirs(plots_dir, exist_ok=True)

        # Save hyperparameter and results info with plots
        final_results = (self._hyperparams, final_eps, test_accuracy)
        save_object(plots_dir, final_results, 'final_results')

        import matplotlib.pyplot as plt  # import here for weird matplotlib+multiprocessign compatibility issue

        # Only create privacy plot if privacy was accumulated
        if len(eps_sequence) > 0:
            # Privacy results
            plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
            plt.grid(True, which='both')
            plt.plot(range(rounds), eps_sequence)
            plt.xlabel('rounds', fontsize=14)
            plt.ylabel('eps', fontsize=14)
            plt.title('Overall (eps,delta)-DP over composition.')
            plt.savefig('{}/privacy-results.png'.format(plots_dir))
            plt.close()

        # Loss results
        plt.figure(num=None, figsize=(8, 6))
        plt.plot(range(len(loss_sequence)), loss_sequence)
        ma_window = 50
        plt.plot(range(ma_window-1, len(loss_sequence)), self.moving_average(loss_sequence, n=ma_window), color='red')
        plt.grid(True, which='both')
        plt.xlabel('\"rounds\"', fontsize=14)
        plt.ylabel('average loss', fontsize=14)
        plt.draw()
        plt.savefig('{}/loss-results.png'.format(plots_dir))
        plt.close()

        # # Gradient clipping results
        # plt.figure(num=None,figsize=(8, 6))
        # plt.plot(norm_sequence)
        # plt.grid(True, which="both")
        # plt.xlabel('rounds',fontsize=14)
        # plt.ylabel('gradient norm',fontsize=14)
        # plt.draw()
        # plt.savefig("{}/dp-lr-norm-results.png".format(plots_dir))
