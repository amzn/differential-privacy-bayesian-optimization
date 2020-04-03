import numpy as np
import pandas as pd

def process_data(raw_data):
    train_lines = raw_data.splitlines()
    num_examples = len(train_lines)
    num_features = 123
    X = np.zeros((num_examples, num_features))
    Y = np.zeros((num_examples, 1))
    for i, line in enumerate(train_lines):
        tokens = line.split()
        #label = tokens[0]
        label = (int(tokens[0]) + 1) / 2  # Change label from {-1,1} to {0,1}
        Y[i] = label
        for token in tokens[1:]:
            index = int(token[:-2]) - 1
            X[i, index] = 1
    return X, Y

def normalize_data(Xtrain, Xtest):
    normalizer = max(np.max(np.linalg.norm(Xtrain, axis=1)),
                 np.max(np.linalg.norm(Xtest, axis=1)))
    Xtrain = Xtrain / normalizer
    Xtest = Xtest / normalizer
    return Xtrain, Xtest

def get_data():
    print("Opening...")
    with open("a1a.train") as f:
        train_raw = f.read()

    with open("a1a.test") as f:
        test_raw = f.read()
    print("done opening.")
    print("Processing...")
    Xtrain, Ytrain = process_data(train_raw)
    Xtest, Ytest = process_data(test_raw)
    print("done processing.")
    print("Normalizing...")
    Xtrain, Xtest = normalize_data(Xtrain, Xtest)
    print("done normalizing.")

    return Xtrain, Ytrain, Xtest, Ytest


from math import exp, sqrt
from scipy.special import erf
from scipy.optimize import root_scalar


def get_eps_AGM(sigma, GS, delta, min_eps=1e-6, max_eps=500, tol=1e-12):
    # Compute the epsilon corresponding to a Gaussian perturbation
    normalized_sigma = sigma / GS

    def Phi(t):
        return 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))

    def get_delta(s, e):
        return Phi(-e * s + 1.0 / (2 * s)) - exp(e) * Phi(-e * s - 1.0 / (2 * s))

    def f(x):
        return get_delta(normalized_sigma, x) - delta

    # Debug output to help set max_eps
    #print('get_eps_AGM | s: %f\tg: %f\tn: %f\td: %f\td0: %f\td+: %f' %
    #      (sigma, GS, normalized_sigma, delta, get_delta(normalized_sigma, min_eps), get_delta(normalized_sigma, max_eps)))

    assert get_delta(normalized_sigma, min_eps) >= delta
    assert get_delta(normalized_sigma, max_eps) <= delta

    sol = root_scalar(f, bracket=[min_eps, max_eps], xtol=tol)
    assert sol.converged

    return sol.root


# ProjSGDClassifier is an sklearn model that needs to be compiled locally
# See README in parent folder
from sklearn.linear_model import ProjSGDClassifier


def dp_proj_sgd(Xtrain, Ytrain, Xtest, Ytest, reg_lambda=0.001, sigma=0.1, delta=1e-6, R=10):
    # Define the model
    clf = ProjSGDClassifier(loss="log", penalty="l2",
                            learning_rate="bolton",
                            alpha=reg_lambda,
                            radius=1.0 / reg_lambda,
                            max_iter=10,
                            verbose=0,
                            fit_intercept=False)
    # print(clf.get_params())

    scores = []
    for r in range(R):
        # Train the model
        clf.fit(Xtrain, Ytrain.ravel())
        # Privatize the model
        Z = sigma * np.random.standard_normal(size=clf.coef_.shape)
        clf.coef_ += Z
        # Evaluate the model accuracy
        score = clf.score(Xtest, Ytest)
        scores.append(score)

    # Evaluate the model privacy
    # Compute the global sensitivity
    m = Xtrain.shape[0]
    GS = 4.0 / (m * reg_lambda)
    epsilon = get_eps_AGM(sigma, GS, delta)

    return np.average(scores), epsilon

def compute_privacy_with_ranges(l_range, s_range, delta=1e-6):
    '''
    Computes the privacy

    X = [c,b]
    '''
    eps = np.zeros((len(l_range), len(s_range)))

    Xtrain, Ytrain, Xtest, Ytest = get_data()
    m = Xtrain.shape[0]

    for i in range(len(l_range)):
        for j in range(len(s_range)):
            index = i * len(s_range) + j

            GS = 4.0 / (m * l_range[i])
            epsilon_value = get_eps_AGM(s_range[j], GS, delta)

            eps[i, j] = epsilon_value

    return np.log(eps)


def compute_outputs_with_ranges(l_range, s_range, R=10, delta=1e-6):
    '''
    Computes the privacy

    X = [c,b]
    '''
    scores = np.zeros((len(l_range), len(s_range)))
    eps = np.zeros((len(l_range), len(s_range)))

    input_matrix = np.zeros((len(l_range) * len(s_range), 2))
    scores_output = np.zeros((len(l_range) * len(s_range), 1))
    eps_output = np.zeros((len(l_range) * len(s_range), 1))

    Xtrain, Ytrain, Xtest, Ytest = get_data()

    for i in range(len(l_range)):
        for j in range(len(s_range)):
            index = i * len(s_range) + j

            score_avg, epsilon_value = dp_proj_sgd(Xtrain, Ytrain, Xtest, Ytest,
                                                   reg_lambda=l_range[i], sigma=s_range[j], delta=delta, R=R)

            scores[i, j] = score_avg
            scores_output[index, 0] = score_avg
            eps[i, j] = epsilon_value
            eps_output[index, 0] = epsilon_value

            ## input
            input_matrix[index, 0] = l_range[i]
            input_matrix[index, 1] = s_range[j]

    return scores, np.log(eps), input_matrix, scores_output, np.log(eps_output)


def predict_outputs_with_ranges(l_range, s_range, model_scores, model_eps):
    '''
    Computes the privacy

    X = [c,b]
    '''
    scores = np.zeros((len(l_range), len(s_range)))
    eps = np.zeros((len(l_range), len(s_range)))

    input_matrix = np.zeros((len(l_range) * len(s_range), 2))
    scores_output = np.zeros((len(l_range) * len(s_range), 1))
    eps_output = np.zeros((len(l_range) * len(s_range), 1))

    for i in range(len(l_range)):
        for j in range(len(s_range)):
            ## input
            index = i * len(s_range) + j

            input_matrix[index, 0] = l_range[i]
            input_matrix[index, 1] = s_range[j]

            ## utility
            score_avg = model_scores.predict_f(input_matrix[index, None])[0]
            scores[i, j] = score_avg
            scores_output[index, 0] = score_avg

            ## privacy
            epsilon_value = model_eps.predict_f(input_matrix[index, None])[0]
            eps[i, j] = epsilon_value
            eps_output[index, 0] = epsilon_value

            ## input
            input_matrix[index, 0] = l_range[i]
            input_matrix[index, 1] = s_range[j]

    return scores, eps, input_matrix, scores_output, eps_output


def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def get_pareto_points(scores):
    pareto = identify_pareto(scores)
    pareto_front = scores[pareto]

    pareto_front_df = pd.DataFrame(pareto_front)
    pareto_front_df.sort_values(0, inplace=True)
    pareto_front = pareto_front_df.values
    x_all = scores[:, 0]
    y_all = scores[:, 1]
    x_pareto = pareto_front[:, 0]
    y_pareto = pareto_front[:, 1]
    return x_pareto, y_pareto, pareto