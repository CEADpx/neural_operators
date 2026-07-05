# neural_operators

This repository implements various neural operators and applies them to parametric PDE forward problems and downstream tasks (Bayesian inversion, operator comparison).

## Environment

Use [neuralop.yml](neuralop.yml) to create the conda environment (`neuralopv2`). Scripts and notebooks were developed on Apple Silicon and Ubuntu 24.04.

```bash
conda env create -f neuralop.yml
conda activate neuralopv2
```

On macOS, if OpenMP errors appear when running PyTorch:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Shared data

Pre-generated training data, trained model weights, and MCMC results for reproducing the survey article are in the Dropbox folder [NeuralOperator_Survey_Shared_Data_June2026](https://www.dropbox.com/scl/fo/co5v2bozvr5y8uv5kc29y/ACKiT1sBBQCTKV2wZYcAIlI?rlkey=agt87l1tf89g967gf8ofe5nik&st=3no4j03k&dl=0).

Each problem's `data/README.md` explains which subfolder to copy locally (Poisson, linear elasticity, hyperelasticity). The Bayesian application READMEs describe additional copies for trained surrogates and MCMC outputs.

## Repository layout

### [src](src)

Shared library code used by `survey_work`:

| Directory | Role |
|-----------|------|
| [src/data](src/data) | Load/process FE and grid data for neural networks |
| [src/pde](src/pde) | Forward PDE models (FEniCSx finite elements) |
| [src/prior](src/prior) | Gaussian prior sampling via $C = L_\Delta^{-2}$ (`PriorSampler`) |
| [src/nn](src/nn) | `DeepONet`, `PCANet`, and `FNO` (PyTorch) |
| [src/mcmc](src/mcmc) | MCMC for Bayesian inversion |
| [src/plotting](src/plotting) | Field and diagnostic plots |

### [survey_work](survey_work)

Notebooks, problem drivers, and results for the neural-operator survey article.

#### [survey_work/problems](survey_work/problems)

Forward problems and neural-operator training:

| Problem | Driver notebook | NOP training |
|---------|-----------------|--------------|
| Poisson | `poisson/Poisson.ipynb` | `{DeepONet,FNO,PCANet}/training_and_testing.ipynb` |
| Linear elasticity | `linear_elasticity/LinearElasticity.ipynb` | same |
| Hyperelasticity | `hyperelasticity/Hyperelasticity.ipynb` | same |

Each problem folder also has `compare_nops/compareNeuralOperators.ipynb` for surrogate comparison.

#### [survey_work/common](survey_work/common)

- [prior_sampler/PriorSampler.ipynb](survey_work/common/prior_sampler/PriorSampler.ipynb) — Gaussian random field sampling (article Fig. on $L_\Delta$ prior)
- [all_model_samples/all_model_samples.ipynb](survey_work/common/all_model_samples/all_model_samples.ipynb) — shared $w$ samples across all three forward models
- `performance_analysis/` — tables and scripts for model performance figures

#### [survey_work/applications](survey_work/applications)

Bayesian inversion of the uncertain coefficient field using the true FE forward map or NOP surrogates:

- [bayesian_inverse_problem_poisson](survey_work/applications/bayesian_inverse_problem_poisson/)
- [bayesian_inverse_problem_linear_elasticity](survey_work/applications/bayesian_inverse_problem_linear_elasticity/)
- [bayesian_inverse_problem_hyperelasticity](survey_work/applications/bayesian_inverse_problem_hyperelasticity/)

Each contains `Generate_GroundTruth.ipynb`, `BayesianInversion.ipynb`, and `prcessed_results/processResults.ipynb`.

#### [survey_work/test_different_implementations_from_public_repositories](survey_work/test_different_implementations_from_public_repositories)

Early comparisons with publicly shared DeepONet/FNO implementations; see [README.md](survey_work/test_different_implementations_from_public_repositories/README.md).

## Citing this work

**Article:** Jha, P. K. (2025). *From Theory to Application: A Practical Introduction to Neural Operators in Scientific Computing.* arXiv:2503.05598 — https://arxiv.org/abs/2503.05598

**Code:** Jha, P. K. (2025). CEADpx: neural_operators (survey25_v1). Zenodo — https://doi.org/10.5281/zenodo.15014505

Article-aligned repository snapshot: Git tag `survey26_v2`.
