from .stochastic_gradient import *
from .proj_sgd_fast import plain_proj_sgd, average_proj_sgd
from .proj_sgd_fast import Hinge
from .proj_sgd_fast import SquaredHinge
from .proj_sgd_fast import Log
from .proj_sgd_fast import ModifiedHuber
from .proj_sgd_fast import SquaredLoss
from .proj_sgd_fast import Huber
from .proj_sgd_fast import EpsilonInsensitive
from .proj_sgd_fast import SquaredEpsilonInsensitive

LEARNING_RATE_TYPES = {"constant": 1, "optimal": 2, "invscaling": 3,
                       "adaptive": 4, "pa1": 5, "pa2": 6, "bolton": 7}

def _prepare_fit_binary_psgd(est, y, i):
    """Initialization for fit_binary.

    Returns y, coef, intercept, average_coef, average_intercept.
    """
    y_i = np.ones(y.shape, dtype=np.float64, order="C")
    y_i[y != est.classes_[i]] = -1.0
    average_intercept = 0
    average_coef = None

    if len(est.classes_) == 2:
        if not est.average:
            coef = est.coef_.ravel()
            intercept = est.intercept_[0]
        else:
            coef = est.standard_coef_.ravel()
            intercept = est.standard_intercept_[0]
            average_coef = est.average_coef_.ravel()
            average_intercept = est.average_intercept_[0]
    else:
        if not est.average:
            coef = est.coef_[i]
            intercept = est.intercept_[i]
        else:
            coef = est.standard_coef_[i]
            intercept = est.standard_intercept_[i]
            average_coef = est.average_coef_[i]
            average_intercept = est.average_intercept_[i]

    return y_i, coef, intercept, average_coef, average_intercept

def fit_binary_psgd(est, i, X, y, alpha, C, learning_rate, max_iter,
               pos_weight, neg_weight, sample_weight, radius=1, validation_mask=None,
               random_state=None):

    y_i, coef, intercept, average_coef, average_intercept = \
        _prepare_fit_binary_psgd(est, y, i)
    assert y_i.shape[0] == y.shape[0] == sample_weight.shape[0]

    random_state = check_random_state(random_state)
    dataset, intercept_decay = make_dataset(
        X, y_i, sample_weight, random_state=random_state)

    penalty_type = est._get_penalty_type(est.penalty)
    learning_rate_type = est._get_learning_rate_type(learning_rate)

    if validation_mask is None:
        validation_mask = est._make_validation_split(y_i)
    classes = np.array([-1, 1], dtype=y_i.dtype)
    validation_score_cb = est._make_validation_score_cb(
        validation_mask, X, y_i, sample_weight, classes=classes)

    # numpy mtrand expects a C long which is a signed 32 bit integer under
    # Windows
    seed = random_state.randint(MAX_INT)

    tol = est.tol if est.tol is not None else -np.inf

    if not est.average:
        result = plain_proj_sgd(coef, intercept, est.loss_function_,
                           penalty_type, alpha, C, est.l1_ratio,
                           dataset, validation_mask, est.early_stopping,
                           validation_score_cb, int(est.n_iter_no_change),
                           max_iter, tol, int(est.fit_intercept),
                           int(est.verbose), int(est.shuffle), seed,
                           pos_weight, neg_weight,
                           learning_rate_type, est.eta0,
                           est.power_t, est.t_, intercept_decay, radius)

    else:
        standard_coef, standard_intercept, average_coef, average_intercept, \
            n_iter_ = average_proj_sgd(coef, intercept, average_coef,
                                  average_intercept, est.loss_function_,
                                  penalty_type, alpha, C, est.l1_ratio,
                                  dataset, validation_mask, est.early_stopping,
                                  validation_score_cb,
                                  int(est.n_iter_no_change), max_iter, tol,
                                  int(est.fit_intercept), int(est.verbose),
                                  int(est.shuffle), seed, pos_weight,
                                  neg_weight, learning_rate_type, est.eta0,
                                  est.power_t, est.t_, intercept_decay,
                                  est.average)

        if len(est.classes_) == 2:
            est.average_intercept_[0] = average_intercept
        else:
            est.average_intercept_[i] = average_intercept

        result = standard_coef, standard_intercept, n_iter_

    return result

class ProjSGDClassifier(SGDClassifier):

    loss_functions = {
        "hinge": (Hinge, 1.0),
        "squared_hinge": (SquaredHinge, 1.0),
        "perceptron": (Hinge, 0.0),
        "log": (Log,),
        "modified_huber": (ModifiedHuber,),
        "squared_loss": (SquaredLoss,),
        "huber": (Huber, DEFAULT_EPSILON),
        "epsilon_insensitive": (EpsilonInsensitive, DEFAULT_EPSILON),
        "squared_epsilon_insensitive": (SquaredEpsilonInsensitive,
                                        DEFAULT_EPSILON),
    }

    def _get_learning_rate_type(self, learning_rate):
        try:
            return LEARNING_RATE_TYPES[learning_rate]
        except KeyError:
            raise ValueError("learning rate %s "
                             "is not supported. " % learning_rate)

    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
                 verbose=0, epsilon=DEFAULT_EPSILON, n_jobs=None,
                 random_state=None, learning_rate="bolton", eta0=0.0,
                 power_t=0.5, early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, class_weight=None, warm_start=False,
                 average=False, radius=1):
        self.radius = radius
        super().__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            shuffle=shuffle, verbose=verbose, epsilon=epsilon, n_jobs=n_jobs,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, class_weight=class_weight,
            warm_start=warm_start, average=average)

    def _fit_binary(self, X, y, alpha, C, sample_weight,
                    learning_rate, max_iter):
        """Fit a binary classifier on X and y. """
        coef, intercept, n_iter_ = fit_binary_psgd(self, 1, X, y, alpha, C,
                                              learning_rate, max_iter,
                                              self._expanded_class_weight[1],
                                              self._expanded_class_weight[0],
                                              sample_weight,
                                              random_state=self.random_state,
                                              radius=self.radius)

        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_

        # need to be 2d
        if self.average > 0:
            if self.average <= self.t_ - 1:
                self.coef_ = self.average_coef_.reshape(1, -1)
                self.intercept_ = self.average_intercept_
            else:
                self.coef_ = self.standard_coef_.reshape(1, -1)
                self.standard_intercept_ = np.atleast_1d(intercept)
                self.intercept_ = self.standard_intercept_
        else:
            self.coef_ = coef.reshape(1, -1)
            # intercept is a float, need to convert it to an array of length 1
            self.intercept_ = np.atleast_1d(intercept)