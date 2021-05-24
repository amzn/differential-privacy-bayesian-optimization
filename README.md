# Automatic Discovery of Privacy-Utility Pareto Fronts

This repo contains the underlying code for all the experiments from the paper: "Automatic Discovery of Privacy-Utility Pareto Fronts" (https://arxiv.org/abs/1905.10862).

To cite this article:

```
@article{avent2020automatic,
  title={Automatic Discovery of Privacy--Utility Pareto Fronts},
  author={Avent, Brendan and Gonzalez, Javier and Diethe, Tom and Paleyes, Andrei and Balle, Borja},
  journal={Proceedings on Privacy Enhancing Technologies},
  volume={2020},
  number={4},
  pages={5--23},
  year={2020},
  publisher={Sciendo}
}
```

## Installation

### Installing from GitHub
You can get the code directly from this repository, and install the necessary dependencies directly from the `requirements.txt` file as follows:

```
git clone https://github.com/amzn/differential-privacy-bayesian-optimization.git
cd differential-privacy-bayesian-optimization
pip install -r requirements.txt
```


### Installing as a local package in a virtual environment
You can install the library via pip, for example in a clean conda environment:

```
conda create -n dpareto python==3.7
conda activate dpareto
pip install -e .
```

Currently supported Python versions: 3.5, 3.6, 3.7

```
pip install -U pip
```
We have verified that this works with pip 18.1 and 20.2.


This project is built in layers, so we'll give a bottom-up explanation for each layer and how to run them.
But first, if you choose to manually install the library, we'll go through the dependencies and what needs to be done before running the code.


## Major dependencies (for manual installation)
 Various portions of the code make use of multiprocessing capabilities of CPU(s) for parallelism, and GPU(s) for efficient model-training. 
 
 - mxnet-cu92mkl (Versions close to this will probably work as well, but be sure to install with the CUDA 'cu' option; MKL isn't necessary, it just helps with performance.)
 - autodp (Version 0.1 is the only one released at the time of writing this.)
 - multiprocessing
 - psutil
 - botorch

**Note: All code will require the root of this project to be in your PATH and/or PYTHONPATH, or set as the "working directory" in your IDE, etc. so that Python can find the dpareto module.**

 We'll explain what needs to be done to get/install autodp, and assume that you have (or can easily get) any of the other dependencies.


### autodp

autodp is an open-source moments accountant implementation in Python, used for computing privacy loss.

It can be installed with pip, and the code is available here:  https://github.com/yuxiangw/autodp


## Project structure

The root of the project contains the following directories, which we'll highlight here and then explain in greater detail in the subsequent sections.
- `dpareto`: This contains the core functionality of the project. It includes:
  - The models that were used for the various datasets (`dpareto/models`).
  - The differentially private optimizers that were used (`dpareto/optimizers`).
  - The various hyperparameter selection frameworks that were implemented to empirically estimate a Pareto front. Namely, random sampling (`dpareto/random_sampling`), grid search (`dpareto/grid_search`), and Bayesian optimization using the hypervolume probability of improvement (`dpareto/hypervolume_improvement`).
  - The various utility methods that were created to aid with the development process (`dpareto/utils`).
- `data`: This contains the code for downloading, processing, and storing the Adult dataset (described below).
- `examples`: This contains simple examples for how the various hyperparameter selection frameworks are used to estimate the Pareto front.
- `experiments`: This contains all the code that was used to carry out the various experiments in the paper.


### Data

This project uses a processed version of the [Adult dataset](https://www.csie.ntu.edu.tw/%7Ecjlin/libsvmtools/datasets/binary.html) as well as the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The Adult dataset is downloaded and processed by running the downloader.py script in the `data/adult` directory from the project root:
```
python data/adult/downloader.py
```
The MNIST dataset is imported automatically using the MXNet Gluon API. 

### Models and Optimizer

The implementations (and usages) of the feedforward neural net code and the parameter optimizer are tightly coupled, so we'll discuss them together.

`dpareto/optimizers/dp_optimizer.py` is an abstract class providing most of the implementation of differentially-private SGD (essentially as detailed in Abadi et al's "Deep Learning with Differential Privacy", with some minor modifications).
`dpareto/optimizers/dp_sgd.py` completes the implementation, and `dpareto/optimizers/dp_adam.py` extends and completes the implementation to create a differentially private variant of the ADAM optimizer.

`dpareto/models/dp_feedforward_net.py` provides the abstract class and implementations of most methods necessary for building a feedforward net.
Its child classes reside in `dpareto/models/mnist` and `dpareto/models/adult` directories each as `base.py`, providing the concrete structure of the network (MLP and SLP respectively) and the dataset-specific functionality for their respective problems.
The `base.py` files in each directory are still abstract classes. The adult directory's `base.py` is then extended to implement logistic regression and SVM models to be trained on the Adult dataset with both the DP SGD and ADAM optimizers. The mnist directory's `base.py` is similarly extended to be trained on the MNIST dataset with the DP SGD and ADAM optimizers.

`dpareto/models/dp_feedforward_net.py` specifies one of the concrete DP optimizers.
The reason that the `dp_feedforward_net.py` code and optimization code are tightly coupled is purely for performance reasons: we have manually created our network and we are manually specifying how batch computations are done (for both the actual results as well as the gradients of the parameters).

#### How to run
Each of the concrete child classes (e.g., `dpareto/models/adult/lr/dp_sgd.py`) contain main functions which can be run (primarily for testing/debugging purposes).
At a high level, the input is a set of hyperparameters and the output is a tuple (privacy, utility) (where privacy is the epsilon differential privacy value and utility is classification accuracy between 0 and 1).
See the main function in those files to get a better idea.

Make sure to run experiments from root of the repo, so that relative paths to Adult dataset work.


### Random Sampling of Hyperparameters

`dpareto/random_sampling/harness.py` provides the abstract class and implementations of most methods necessary for performing random sampling on one of these concrete dp_feedforward_net problems.

For a simple example of how this can be used, see `examples/random_sampling.py`.

The primary inputs here are:
 - The _distribution_ of hyperparameters to sample from for the specific problem.
 - The number of instances; i.e., how many hyperparameter samples to take.

The output is the full set of results -- that is: a list of tuples where the first item in each is the input (a single random setting of hyperparameters), and the second and third items respectively are the privacy and utility that resulted from an execution of that setting of hyperparameters.

Since these instances are fully independent, we (optionally) take advantage of parallel processing by also specifying the number of workers to use to compute all the instances.
We can also provide command line arguments to override the passed-in number of instances, number of workers, and device type (CPU or GPU).
See the main function in either child class file to get a better idea.

### Grid Search of Hyperparameters

`dpareto/grid_search/harness.py` provides the abstract class and implementations of most methods necessary for performing random sampling on one of these concrete dp_feedforward_net problems.

Everything here is analogous to the random sampling discussed above, and a simple example of its use is in `examples/grid_search.py`.

### Computing and Optimizing the Pareto Front

`dpareto/hypervolume_improvement/harness.py` provides the abstract class and implementations of most methods necessary for computing and optimizing the pareto front.

A simple example of its use is in `examples/hypervolume_improvement.py`.

There are several options for input here currently, but the primary idea is:
 1) Pass in some initial points to kick-start the GP models for privacy and accuracy. This requires at least one run of random sampling. In our experiments we used 16 initial points.
 2) Provide the "anti-ideal" point -- i.e., the reference point from which to compute the hypervolume from (think: top right point, and we want to optimize towards the bottom left). In our experiments we used [10, 0.999] based on the ranges of the output domain.
 3) Provide the optimization domain -- i.e., the range of valid values for each hyperparameter that the optimizer should consider.
 4) The number of points that the Bayesian optimizer should test. Note: this is the time-consuming portion, particularly because we can't currently do it in parallel.
The output is saved to disk, as:
 - The set of hyperparameters suggested by the BO framework at each iteration
 - Pareto front plots (the plot from the initial data, as well as the new plot for each point tested by the Bayesian optimizer).
 - The full state of the hypervolume improvement class object at each iteration, to allow extraction of any results at a later time.
   - Due to a known bug related to the unpickling of an object which contains use of the multiprocessing package, extra steps must be taken to unpickle these object-state results. `experiments/scripts/picky_unpickler.py` was created to enable the extraction of specific results from these objects -- look through that file to see what is done and how.

### Paper Experiments
The runnable code for all the experiments in the paper is located in the `experiments` directory.
The `experiments/output_perturbation` subdirectory contains the relevant code for the "illustrative example" of training a logistic regression model using output perturbation.
The `experiments/svt` subdirectory contains the relevant code for the "illustrative example" of the sparse vector technique algorithm.
The `experiments/adult` and `experiments/mnist` subdirectories contain the code for the experiments run on the various models trained on the Adult and MNIST datasets.


#### Example
Let's review an example of running a full experiment for an already provided algorithm. We will use Adult dataset/Logistic Regression/Adam optimizer. Switch to project's root folder, make sure dependencies above are installed, and update PATH and PYTHONPATH variables as needed. For example, this verifies that we have botorch installed:

```
pip freeze | grep -i botorch
```

And this adds project's root folder to PATH, assuming we are already in that folder:
```
export PATH=$PATH:$(pwd)
```

As noted above, having at least one run of random sampling is required, so that we have some initial data to bootstrap the GP model. Let's do this:
```
python experiments/adult/dp_adult_lr_adam/random_sampling.py --workers 4 --instances 128
```
This says we will use 4 CPUs concurrently on the machine, and that we would like to collect 128 random points.

Once this finishes, we can run the Bayesian optimization routine. It requires only one parameter, the path to the random sampling results, serialized with Python's pickle module. You can find it by browsing "experiments/adult/dp_adult_lr_adam/results" folder. Note that failed runs can also create results folders, so make sure to pick the correct file. Result folders are named according to corresponding timestamps.

```
python experiments/adult/dp_adult_lr_adam/hypervolume_improvement.py --init-data-file experiments/adult/dp_adult_lr_adam/results/random_sampling_results/1578475241956/full_results.pkl
```

Alternatively we could just edit the executed file directly, by adding this line:
```
initial_data_options['filename'] = current_dir + '/results/random_sampling_results/1578475241956/full_results.pkl'
```

Once this run finishes, results will be available in `experiments/adult/dp_adult_lr_adam/results` folder as well.
