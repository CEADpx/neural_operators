This folder includes notebook files to generate the synthetic data and solve the Bayesian inference problem for the linear elasticity problem. The implementation allows either using the ``true" forward model (upto FE discretization error) or the surrogates based on neural operators. 

## Copy shared data before running the notebooks in this folder
In case:
    - did not generate data in `survey_work/problems/linear_elasticity/data`; or
    - did not train the neural operators in `survey_work/problems/linear_elasticity/[DeepONet, PCANet, FNO]` and therefore the folders do not have `Results` folder containing the trained model weights (e.g., `survey_work/problems/linear_elasticity/DeepONet/Results/model.pkl`)
please consider copying the folders from shared Dropbox folder [NeuralOperator_Survey_Shared_Data_March2025](https://www.dropbox.com/scl/fo/5dg02otewg7j0bt7rhkuf/AOfAAc2SaWOgO-Yg25IlTXs?rlkey=t900geej8y8z327y5f8wu4yc9&st=t9c8qimk&dl=0) into the relevant directories. To be precise, copy
    - `NeuralOperator_Survey_Shared_Data_March2025/survey_work/problems/linear_elasticity/data/` contents into the local folder `survey_work/problems/linear_elasticity/data`; and
    - `NeuralOperator_Survey_Shared_Data_March2025/survey_work/problems/linear_elasticity/[DeepONet, PCANet, FNO]/Results/model.pkl` into the local folder `survey_work/problems/linear_elasticity/[DeepONet, PCANet, FNO]/Results/`

## Generate_GroundTruth.ipynb
This notebook generates synthetic data.

## BayesianInversion.ipynb
Solves the Bayesian inference problem.

## Results 
The directory contains the results of the above two notebooks.

## processed_results
The folder contains the notebook to process the Bayesian inference results from the ``true" and surrogate models. 

The results in the survey paper are based on the results in the following four folders:
- `FINAL_mcmc_results_n_samples_10000_n_burnin_500_pcn_beta_0.150_sigma_2.865e-04`
- `FINAL_mcmc_results_n_samples_10000_n_burnin_500_pcn_beta_0.150_sigma_2.865e-04_surrogate_DeepONet`
- `FINAL_mcmc_results_n_samples_10000_n_burnin_500_pcn_beta_0.150_sigma_2.865e-04_surrogate_PCANet`
- `FINAL_mcmc_results_n_samples_10000_n_burnin_500_pcn_beta_0.150_sigma_2.865e-04_surrogate_FNO`

These folders can be found in the subdirectory `survey_work/applications/bayesian_inverse_problem_linear_elasticity/Results/` [NeuralOperator_Survey_Shared_Data_March2025](https://www.dropbox.com/scl/fo/5dg02otewg7j0bt7rhkuf/AOfAAc2SaWOgO-Yg25IlTXs?rlkey=t900geej8y8z327y5f8wu4yc9&st=t9c8qimk&dl=0).
