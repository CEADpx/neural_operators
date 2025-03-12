# neural_operators

This repository aims to implement (or interface with existing libraries) various neural operators and apply them to solve a few parametric linear PDEs. 

## Content description

We include [neuralop.yml](neuralop.yml) file that can be used to setup the conda environment. The scripts and notebooks are mostly used in Ubuntu 24.04 and hopefully the dependencies mentioned in the script will be sufficient to run the files.

Next, some key directories are described. 

### [src](src)
This directory contains various methods and classes used in `survey_work`. It tries to make the problems object-oriented so that same functions are not defined from scratch. Specifically, the directory contains following methods:

1. [src/data](src/data): Data related methods to process data and make it ready for neural networks.
1. [src/pde](src/pde): PDE related methods to model the PDE-based forward problem. We rely on Fenics for finite element method.
1. [src/prior](src/prior): We implement Gaussian-measure based prior for generating samples of random fields in function space. It implements the Gaussian prior based on the inverse of elliptic operator. 
1. [src/nn](src/nn): Implementation of three key neural operators `DeepONet`, `PCANet`, and `FNO` using torch library.
1. [src/mcmc](src/mcmc): MCMC implementation. The implementation is done from scratch, and as such it does not depend on any external library for MCMC. 
1. [src/plotting](src/plotting): Several functions for plotting fields on finite element unstructured and grid mesh and point and scatter data are included. 

### [survey_work](survey_work)
This directory contains the scripts, notebooks, and results for the neural oeprator survey article. Key files and directories in this directory are as follows

#### [survey_work/problems](survey_work/problems) 
Neural operators are applied to Poisson and Linear Elasticity problems. This directory contains the notebooks used to generate data and train and test three neural operators, `DeepONet`, `PCANet`, and `FNO`.

#### [survey_work/applications](survey_work/applications) 

Neural oeprators as a surrogate of the forward model is applied to [Bayesian inverse](survey_work/applications/bayesian_inverse_problem) and [topology optimization](survey_work/applications/topology_optimization) problems. The subdirectories contain implementation and results for these two applications. 

#### [survey_work/test_different_implementations_from_public_repositories](survey_work/test_different_implementations_from_public_repositories)
The contents in this directory were extremely useful in testing the neural operator implementations. Different versions of implementations available in publicly-shared repositories are considered as a first step in our goal to survey the neural operator techniques. The [README.md](survey_work/test_different_implementations_from_public_repositories/README.md) inside this directory provides further details. Based on the information gained, three neural operator models `DeepONet`, `PCANet`, and `FNO` are implemented using `torch` library and as mentioned earlier the implementations are in [src](src) directory.


## Citing this work

Code:
<a id="1">[1]</a> 
Jha, P. K. (2025). 
CEADpx: neural_operators (survey25_v1).
Zenodo, link: https://doi.org/10.5281/zenodo.15014505

Article:
<a id="1">[2]</a> 
Jha, P. K. (2025). 
From Theory to Application: A Practical Introduction to Neural Operators in Scientific Computing.
arXiv, arXiv:2503.05598, link: https://arxiv.org/abs/2503.05598
 
