# Hyperelasticity

2D compressible hyperelasticity (Neo-Hookean energy) with the same setup as `linear_elasticity/`:

- Unit square mesh (`50×50` triangles, P1)
- Uncertain Young's modulus from log-normal prior on scalar `Vm`
- Left edge Dirichlet clamp `u = 0`
- Body force zero, uniform traction `(tx, ty)` on the exterior boundary
- Mixed FEniCSx vector layout for displacement

## Files

| File | Purpose |
|------|---------|
| `hyperelasticityModel.py` | Forward model (`PDEModel` API, Newton solve) |
| `Hyperelasticity.ipynb` | Prior tests, data generation, FNO grid export |
| `data/` | Generated `.npz` samples (after running the notebook) |

## Run

```bash
conda activate neuralopv2
jupyter lab Hyperelasticity.ipynb
```

Validation: `python setup/fenicsx_test/test_hyperelasticity_model.py`

The implementation follows the stable compressible Neo-Hookean energy
(`(μ/2)(I₁ - 3 - 2 ln J) + (λ/2)(ln J)²`) used in the CEADpx phase-field demo,
reduced to 2D with the same BC/traction pattern as linear elasticity.

Neural-operator training notebooks live under `{DeepONet,FNO,PCANet}/` with `data_prefix = 'Hyperelasticity'`.
