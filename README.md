# Instructions

**Instructions to run machine learning application**

# Setup

Install

* Matlab

* Python

* Tensorflow

* Create folder input_data and add input .csv files -> outdated

# Python Tensorflow

* Run in matlab `read_data.m`

for data extraction from output.csv

* Run in matlab `ml_data_format.m`

for data formating into inputs and targets, saved as inputs.csv and targets.csv

* Run `python dataset_extractor.py`

data extraction from inputs.csv and targets.csv,

splitting into test, validate and train datasets & targets

saved as pickle

* Run `python print_dataset.py`

performs data checks

prints histos from targets (5 parameters)

should: prints histos from inputs (5 parameters)

* Run `python first_perc_one_hidden_layer.py`

Machine Learning Code

should: work

should: extract weights & bias so that single inputs can be analyzed.

# Matlab

* Run in matlab `ml_matlab.m`

for data extraction from output.csv

this script calls:

* `read_data.m`

for data extraction from output.csv

* `make_data.m`

* `results_print.m`

* `benchmark_check.m`

* `analyze_ml_output.m`

# Author

Christoph Bernkopf

Masterstudent Technical Physics

TU Vienna, HEPHY, CERN
