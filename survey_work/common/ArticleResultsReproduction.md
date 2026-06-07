# Article reproduction checklist (FEniCSx / `neuralopv2`)

Run notebooks in **`neuralopv2`**. Numerical outputs will differ slightly from legacy FEniCS artifacts; parameters below match the article setup.

## Verified parameters (article-aligned)

| Parameter | Poisson | Linear elasticity |
|-----------|---------|-------------------|
| Mesh `nx`, `ny` | 50, 50 | 50, 50 |
| `fe_order` | 1 | 1 |
| Prior `prior_ac`, `prior_cc` | 0.005, 0.2 | 0.005, 0.2 |
| Log-normal `α_m`, `β_m` | 1, 0 | 100, 1000 |
| Training samples | 5000 | 5000 |
| NOP train / test split | 3500 / 1000 | 3500 / 1000 |
| FNO grid | 51×51 | 51×51 |
| Observation grid | 16×16, linspace(0.05, 0.95) | same |
| MCMC `n_samples` / `n_burnin` | 10000 / 500 | 10000 / 500 |
| PCN `pcn_beta` | 0.2 | 0.15 |
| Noise `σ` | 0.05 × mean(`u_obs`) | 0.01 × mean(`u_obs`) |

Vector displacement data use **mixed FEniCSx layout** `[ux(v₀), uy(v₀), ux(v₁), …]` everywhere (mesh, FNO grid, observations).

## Recommended run order

1. **Problem drivers** — generate FE training data and FNO grids:
   - `survey_work/problems/poisson/Poisson.ipynb`
   - `survey_work/problems/linear_elasticity/LinearElasticity.ipynb`

2. **Neural operators** — train surrogates (after step 1):
   - `{Poisson,LinearElasticity}/{DeepONet,FNO,PCANet}/training_and_testing.ipynb`

3. **Bayesian ground truth**:
   - `survey_work/applications/bayesian_inverse_problem_poisson/Generate_GroundTruth.ipynb`
   - `survey_work/applications/bayesian_inverse_problem_linear_elasticity/Generate_GroundTruth.ipynb`

4. **Bayesian inversion (MCMC)**:
   - `survey_work/applications/bayesian_inverse_problem_poisson/BayesianInversion.ipynb`
   - `survey_work/applications/bayesian_inverse_problem_linear_elasticity/BayesianInversion.ipynb`

## Flags to set before a full regeneration

### `Poisson.ipynb` / `LinearElasticity.ipynb`

```python
generate_data = True       # 5000 FE samples → data/*_samples.npz
generate_fno_data = True   # FNO grid data → data/*_FNO_samples.npz
```

Leave `False` to load existing `.npz` files.

### `Generate_GroundTruth.ipynb` (both problems)

First run (or if `w_true.npy` missing):

```python
save_w = True
```

Then write Bayesian observation bundle:

```python
save_for_bayesian_grid_data = True   # → Results/ground_truth/data.npz
```

### `BayesianInversion.ipynb` (both problems)

Publication defaults (already set):

```python
run_small_test = False   # True → 100/20 samples, test_mcmc_ prefix (smoke test only)
load_surrogate = True
```

Each MCMC section has `run_mcmc = True`. Set `run_mcmc = False` in a section to skip re-sampling and load an existing tracer from `Results/mcmc_*`.

## Smoke tests

For quick checks without full MCMC cost:

```python
run_small_test = True   # 100 samples, 20 burn-in; only DeepONet surrogate loaded
```

## Environment

```bash
conda activate neuralopv2
# macOS OpenMP (if needed):
export KMP_DUPLICATE_LIB_OK=TRUE
```
