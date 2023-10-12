## SCRIPTS

In this directory you find all the scripts to process the data generated on Colosseum/SCOPE and produce the results (intermediate and final).

### Tree

``` 
.
├── README.md
├── action-steering-plotting.py
├── action-steering-processing.py
├── analysis-explora.py
├── dt_train.py
├── extract_shap_values.py
├── motivation_computing_times_gpus.py
├── motivation_process-shap-computing-times.py
├── requirements.txt
├── shap_vs_dt.py
├── utils_attr_graphs.py
├── utils_db_process.py
├── utils_experiment_list.py
├── utils_experiment_list_action_steering.py
├── utils_generic_functions.py
├── utils_shap.py
└── utils_shap_plotting.py

```

### Description

The scripts are divided into two main groups:
1. Libraries with utility functions (their name start with `utils`).
1. Libraries to process the data and generate the results. 

Specifically:

1. Libraries with functions:

	  - `utils_shap_plotting.py`: functions to visualize SHAP explanations on the DRL inputs of the experiments
	  - `utils_shap.py`: functions to extract SHAP values from the DRL agents applied in the experiments
	  - `utils_db_process.py`: functions to process the pkl/log files that are the generated with SCOPE
	  - `utils_generic_functions.py`: generic functions
	  - `utils_attr_graphs.py`: functions for the attributed graphs
	  - `utils_experiment_list.py`: configuration of experiments useful for the motivation (`extract_shap_values.py`, `shap_vs_dt.py`) and the core EXPLORA analysis (`analysis-explora.py`)
	  - `utils_experiment_list_action_steering.py`: configuration of experiments useful for action steering analysis (`action-steering-processing.py`, `action-steering-plotting.py`)
  
2. Code for analysis:
	  - Motivation, i.e., why SHAP or Decision Trees (DTs) do not work properly for the purpose (execute in the exact same order):
	    - `dt_train.py`: trains XGBoost DTs on the offline dataset for the two agents; saves the model for later use in `shap_vs_dt.py` 
	    - `extract_shap_values.py`: computes the SHAP values from the DRL agent directly and stores all the results in [shap_explanations/drl_agent](shap_explanations/drl_agent) - Careful with this script: several experiments take hours to complete; make sure you set `export TF_ENABLE_ONEDNN_OPTS=0` and have GPU enabled
	    - `shap_vs_dt.py`: script to generate and export the results included in the motivation *Section 3.2* of the paper: SHAP results and DT accuracy
	    - `motivation_process-shap-computing-times.py`: computes computing times for configurations of the `embb-tfr2` (HT) and `urllc-tfr2` (LL) agents
	    - `motivation_computing_times_gpus.py`: computes computing times on the two GPUs for all configurations of the `urllc-tfr2`
	  - *EXPLORA* core part:
	    - `analysis-explora.py`: performs the major operations: creates attributed graphs, process them, synthesizes explanations from the attributed graphs, and build DTs on top of the generated explanations. The script generates the results in *Section 6.2* and *Appendix C* of the paper.
	  - Action steering with intents:
	    - `action-steering-processing.py`: processes the experiments performed on Colosseum/SCOPE with the *EXPLORA* xApp that builds at runtime the attributed graphs and implements the 3 strategies for action replacement (Section 5.2 of the paper); the script produces `.npy` files as output of the process that are the KPIs observed upon implementing the replacement strategies
	    - `action-steering-plotting.py`: plots the intermediate results generated by the above script and generates the results of *Section 6.3* and *Appendix D* of the paper

