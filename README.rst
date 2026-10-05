psyc-pathology-main
----------------

**Brief Description**

This repository contains data, MATLAB code, and Python scripts associated with our paper, Dissociating volatility and stochasticity reveals transdiagnostic computational signatures of psychopathology. 

**File organization**::

  psyc-pathology-main/
  ├── mat_data/
  │   ├── experiment_1/
  │   │   ├── critical_value_grouping_fascore_bird.mat
  │   │   │   (theory-guided grouping results for the bird task, including grouped learning rates,
  │   │   │   factor-score summaries, and covariate analyses)
  │   │   ├── data_bird.mat
  │   │   │   (preprocessed bird-task trial data)
  │   │   ├── fa_score_ex2_factors2_bird.mat
  │   │   │   (two-factor EFA results for the bird task)
  │   │   ├── factor_glm_mn_lr_bird.mat
  │   │   │   (GLM results of factors ~ lr effects)
  │   │   ├── mn_lr_glm_factor_bird.mat
  │   │   │   (GLM results of lr effects ~ factors)
  │   │   ├── model_neutral_bird.mat
  │   │   │   (model-neutral learning-rate estimates for the bird task)
  │   │   ├── model_neutral_pilot1.mat
  │   │   │   (pilot dataset 1 model-neutral learning-rate estimates for the bird task)
  │   │   └── model_neutral_pilot2.mat
  │   │       (pilot dataset 2 model-neutral learning-rate estimates for the bird task)
  │   │
  │   ├── experiment_2/
  │   │   ├── data_sealion_aligned.mat
  │   │   │   (preprocessed aligned sea lion task data)
  │   │   ├── data_turtle_aligned.mat
  │   │   │   (preprocessed aligned turtle task data)
  │   │   ├── distrHMM_rho_fit_params_sealion_aligned.mat
  │   │   │   (distributed-HMM fitted parameter estimates for the aligned sea lion task)
  │   │   ├── distrHMM_rho_fit_params_turtle_aligned.mat
  │   │   │   (distributed-HMM fitted parameter estimates for the aligned turtle task)
  │   │   ├── distrHMM_rho_fit_sealion.mat
  │   │   │   (distributed-HMM fit results for the sea lion task, used for model recovery analyses)
  │   │   ├── fa_score_pooled_ex2_factors2.mat
  │   │   │   (pooled factor scores across experiment 2 tasks)
  │   │   ├── factor_glm_distr_lr_sealion_aligned.mat
  │   │   │   (GLM results relating factor scores and distributed-HMM learning-rate effects in sea lion)
  │   │   ├── factor_glm_distr_lr_supp_sealion_aligned.mat
  │   │   │   (supplementary GLM results for sea lion)
  │   │   ├── factor_glm_distr_lr_supp_turtle_aligned.mat
  │   │   │   (supplementary GLM results for turtle)
  │   │   ├── factor_glm_distr_lr_turtle_aligned.mat
  │   │   │   (GLM results relating factor scores and distributed-HMM learning-rate effects in turtle)
  │   │   ├── factor_glm_distr_lr_valence.mat
  │   │   │   (GLM results for distributed-HMM learning-rate valence analyses)
  │   │   ├── factor_glm_lr_valence.mat
  │   │   │   (GLM results for model-neutral learning-rate valence analyses)
  │   │   └── hidden_state.mat
  │   │       (hidden-state trajectories used for experiment 2 analyses)
  │   │
  │   └── experiment_sim/
  │       ├── data_sim_binary_distrHMM_rho.mat
  │       │   (simulated data for distributed-HMM analyses)
  │       ├── distrHMM_rho_fit_sim/
  │       │   (folder containing simulation-based distributed-HMM fit outputs)
  │       ├── distrHMM_rho_fit_sim_params.mat
  │       │   (simulated/recovered parameter sets for parameter recovery analysis)
  │       ├── distrHMM_rho_fit_sim.mat
  │       │   (distributed-HMM simulation fit results)
  │       └── sim_lesioned.mat
  │           (simulation outputs for lesioned-model analyses)
  │
  ├── matlab_code/
  │   ├── code_exp1/
  │   │   ├── critical_value_grouping_fascore.m
  │   │   │   (theory-guided classification analysis for bird-task factor scores)
  │   │   ├── demo2tbl.m
  │   │   │   (generate demographic summary tables)
  │   │   ├── factor_glm_mn_lr_supp.m
  │   │   │   (supplementary GLM analyses for bird-task factor scores)
  │   │   ├── factor_glm_mn_lr.m
  │   │   │   (main GLM analysis relating bird-task factor scores and model-neutral LR effects)
  │   │   ├── factor_score.m
  │   │   │   (factor-score computation / analysis for experiment 1)
  │   │   ├── get_data.m
  │   │   │   (load and preprocess bird-task data)
  │   │   ├── mn_lr_glm_factor.m
  │   │   │   (GLM analysis relating model-neutral learning rates and factor scores)
  │   │   └── model_neutral.m
  │   │       (compute model-neutral learning-rate measures)
  │   │
  │   ├── code_exp2/
  │   │   ├── cbm/
  │   │   │   (CBM fitting code and dependencies)
  │   │   ├── distrHMM_model.m
  │   │   │   (defines the distributed-HMM model)
  │   │   ├── distrHMM_recovery_lr.m
  │   │   │   (learning-rate recovery analysis for distributed-HMM)
  │   │   ├── distrHMM_rho_fit.m
  │   │   │   (fit distributed-HMM models)
  │   │   ├── distrHMM_rho_recovery.m
  │   │   │   (parameter recovery analysis for distributed-HMM)
  │   │   ├── factor_glm_distr_lr_supp.m
  │   │   │   (supplementary GLM analyses for experiment 2)
  │   │   ├── factor_glm_distr_lr.m
  │   │   │   (main GLM analysis relating factor scores and distributed-HMM LR effects)
  │   │   ├── factor_glm_lr_valence.m
  │   │   │   (GLM analysis for valence-related learning-rate effects)
  │   │   ├── factor_score_pooled.m
  │   │   │   (pooled factor-score analysis across Sea Lion and Turtle tasks)
  │   │   ├── get_data.m
  │   │   │   (load and preprocess experiment 2 data)
  │   │   └── response_model.m
  │   │       (response model definition)
  │   │
  │   ├── tools/
  │   │   (shared helper functions used across analyses)
  │   └── stats_table.m
  │       (generate manuscript statistics tables)
  │
  ├── python_code/
  │   ├── fig_toolbox.py
  │   │   (custom plotting utilities used for manuscript figures)
  │   └── figures.ipynb
  │       (notebook to reproduce the main manuscript figures)
  │    
  ├── saved_figures/
  │   └── (stores figures generated from figures.ipynb and MATLAB plotting scripts)
  │
  └── README.md

**Prerequisites**

- MATLAB R2023b
- Python 3.11.5

**Installation & Setup**

Clone this repository:

  git clone https://github.com/piraylab/psyc-pathology-main.git

  cd psyc-pathology-main

Install Python dependencies:

  pip install -r requirements.txt

**Data Processing Workflow**

1. MATLAB analysis: Use scripts in matlab_code to analyze data stored in mat_data.
2. Python figure generation: Use scripts in python_code to visualize results and save figures to saved_figures.

**Citation**

If you find this work useful, please cite our paper: 
Fang, X., & Piray, P. (2026). Dissociating volatility and stochasticity reveals transdiagnostic computational signatures of psychopathology. bioRxiv : the preprint server for biology, 2026.05.22.727329. https://doi.org/10.64898/2026.05.22.727329
