# neural_operators_survey

This repository aims to implement (or interface with existing libraries) various neural operators and apply them to solve a few parametric linear PDEs. 

## neural_models
To host different neural network/operator models.

## problems
Files for the two model problems.

## utilities
Some utility functions.

## neuralop.yml
Dependencies are listed in this file, and it can be used to create a conda environment to test various scripts in this repository. 

## base_example_to_try_different_implementations
In this folder, base example of Poisson's equation on triangle domain with notch is considered and the map from Dirichlet boundary condition to the PDE solution is learned using DeepONet. **Different versions of implementations available in various repositories are considered as a first step in this survey.** The [README.md](base_example_to_try_different_implementations/README.md) provides further details. 