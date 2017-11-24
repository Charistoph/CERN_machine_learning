# Instructions

**Instructions to run machine learning application**

* Run in matlab `read_data.m`
for data extraction from output.csv

* Run in matlab `ml_data_format.m`
for data formating into inputs and targets, saved as inputs.csv and targets.csv

* Run `p2 dataset_extractor.py`
data extraction from inputs.csv and targets.csv,
splitting into test, validate and train datasets & targets
saved as pickle

* Run `p2 print_dataset.py`
performs data checks
prints histos from targets (5 parameters)
should: prints histos from inputs (5 parameters)

* Run `p2 first_perc_one_hidden_layer.py`
Machine Learning Code
should: work
should: extract weights & bias so that single inputs can be analyzed.
