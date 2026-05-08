PF-HMM
----------------

**Brief Description**

This repository contains data, MATLAB code, and Python scripts associated with our theoretical paper, Inferring the causes of noise from binary outcomes:
A normative theory of learning under uncertainty. The repository is structured to clearly separate preprocessing, data analysis, and figure generation.

**File organization**::

  PF-HMM/
    ├── mat_data/
    │   └── experiment_sealion/
    │     └── data_sealion.mat (preprocessed trial data)
    │     └── hidden_state.mat (the hidden states of the actual time series)
    │     └── reward_stimuli.pkl (the reward location (observation) of the time series)
    │   └──experiment_turtle/
    │   └── experiment_binA/
    │   └── experiment_binB/
    │   └── experiment_sim/
    │     └── timeseries100.mat (the generated timeseries with 100 trials used in plotting fig2a)
    ├── matlab_code/
    │   └── binary_hgf_fit.m (fits the binary HGF model using binary_hgf_model.m)
    │   └── cbm/ (folder contains cbm fitting code)
    │   └── tools/ (additional tool code)
    │   └── hmm.m (defines the HMM model)
    │   └── hmm_rho_fit.m (run the script to fit HMM)
    │   └── model_comparison.m (run the script for model comparison results between HMM and HMM-beta)
    │   └── hmm_rho_recovery.m (run the script for HMM param recovery analysis)
    │   └── other_bmc.m (run the script for model comparison results between PF-HMM, binary HGF, and PHA)
    │   └── pearcehall_fit.m (fits the Pearce-Hall model using pearcehall_model.m)
    │   └── pfhmm.m (defines the PF-HMM model)
    │   └── pfhmm_sim.m (run the script to simulate PF and HMM)
    │   └── pfhmm_rho_fit.m (run the script to fit PF-HMM with preservation)
    │   └── pfhmm_rho_recovery.m (run the script for PF-HMM param recovery analysis)
    │   └── pfhmm_rt_analysis.m (run the script for response time analysis)
    │   └── response_model.m (defines the response model)
    │   └── stats_table.m (produce tables)
    ├── python_code/
    │   └── figures.ipynb (run the script to reproduce the main figures)
    ├── saved_figures/
    │   └── (store figures saved from figures.ipynb, hgf_plot.m, hmm_rho_recovery.m, and pfhmm_rho_recovery.m)
    └── README.md

**Prerequisites**

- MATLAB R2023b
- Python 3.11.5

**Installation & Setup**

Clone this repository:

  git clone https://github.com/piraylab/PF-HMM.git

  cd PF-HMM

Install Python dependencies:

  pip install -r requirements.txt

**Data Processing Workflow**

1. MATLAB analysis: Use scripts in matlab_code to analyze data stored in mat_data.
2. Python figure generation: Use scripts in python_code to visualize results and save figures to saved_figures.

**Citation**

If you find this work useful, please cite our paper: https://osf.io/preprints/osf/vuc5g_v1

experiment_binA from:
Piray, P., Ly, V., Roelofs, K., Cools, R., & Toni, I. (2019). Emotionally aversive cues suppress neural systems underlying optimal learning in socially anxious individuals. Journal of Neuroscience, 1394–18. https://doi.org/10.1523/JNEUROSCI.1394-18.2018

experiment_binB from: 
Jang, A. I., Nassar, M. R., Dillon, D. G., & Frank, M. J. (2019). Positive reward prediction errors during decision-making strengthen memory encoding. Nature Human Behaviour, 3(7), Article 7. https://doi.org/10.1038/s41562-019-0597-3
