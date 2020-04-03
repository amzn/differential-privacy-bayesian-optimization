import mxnet as mx
from mxnet import nd, gluon
import numpy as np

import dpareto.data.adult.importer as importer
from dpareto.models.adult.base import AdultBase
from dpareto.utils.lot_sampler import LotSampler


class DpAdultSvmSgd(AdultBase):
    def __init__(self, hyperparams, options={}):
        super(DpAdultSvmSgd, self).__init__(hyperparams, options)

        self._input_layer = 123
        self._hidden_layers = []
        self._output_layer = 1

        self._name = 'dp_adult_svm_sgd'

    def _load_data(self):
        xTrain, yTrain, xTest, yTest = importer.import_adult_dataset(self._data_ctx)
        yTrain = yTrain * 2 - 1
        yTest = yTest * 2 - 1

        num_training_examples = len(yTrain)
        num_testing_examples = len(yTest)

        if self._verbose:
            print("Loading...")
        train_data = gluon.data.ArrayDataset(xTrain, yTrain)
        train_data_lot_iterator = gluon.data.DataLoader(train_data,
                                                        batch_sampler=LotSampler(self._lot_size, num_training_examples))
        train_data_eval_iterator = gluon.data.DataLoader(train_data, batch_size=self._lot_size)
        test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(xTest, yTest),
                                          batch_size=self._lot_size, shuffle=True)
        if self._verbose:
            print("done loading.")

        return num_training_examples, num_testing_examples, train_data_lot_iterator, train_data_eval_iterator, test_data

    @staticmethod
    def _get_loss_func():
        return mx.gluon.loss.HingeLoss()

    def _evaluate_accuracy(self, data_iterator):
        num_correct = 0
        total = 0
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(self._model_ctx).reshape((-1, 1, self._input_layer))
            label = label.as_in_context(self._model_ctx)

            output = self._net(data, self._params, training_mode=False)
            prediction = (output > 0.0) * 2 - 1

            num_correct += nd.sum(prediction == label)
            total += len(label)

        return num_correct.asscalar() / total


# Small run for testing purposes.
def main():
    print("Running dp_adult_svm_sgd's main().")

    # Fixing some combo of these random seeds is useful for debugging. I'll fix them all to be safe.
    import random
    random_seed = 112358
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)

    # Some default hyperparameter setting.
    hyperparams = {'epochs': 1,
                   'lot_size': 64,
                   'lr': 0.05,
                   'l2_clipping_bound': 2.0,
                   'z': 1.0,
                   'fixed_delta': 1e-5}
    print("hyperparams: {}".format(hyperparams))

    options = {'use_gpu': False, 'verbose': True, 'accumulate_privacy': True, 'debugging': True}

    instance = DpAdultSvmSgd(hyperparams, options)
    priv, acc = instance.run()
    print("Instance privacy: {}".format((priv, hyperparams['fixed_delta'])))
    print("Instance accuracy: {}".format(acc))


if __name__ == "__main__":
    main()
