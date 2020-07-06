# Comparison of machine learning  algorithm performances
This repository contains the source code used to compare the performance of different machine learning algorithms for classification and regression.

The performances of 10 classification algorithms were compared on 10 different classification tasks, and the performances of 10 regression algorithms were compared on 16 different regression tasks.

For each dataset, pre-processing and basic feature engineering was performed.
Automatic hyper-parameter tuning has been used for each algorithm and dataset with [Tree-structured Parzen estimators](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf). The use of automatic parameter-tuning make it possible to ensure that the same effort is invested for the tuning of each algorithm.

Datasets are automatically downlowded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

## Results

### Regression tasks

![Critical Difference Diagram for Regression tasks](results/cd-diagram-regression.png?raw=true)

### Classification tasks

![Critical Difference Diagram for Classification tasks](results/cd-diagram-classification.png?raw=true)

### Regression tasks with big datasets

![Critical Difference Diagram for Classification tasks with bigger datasets](results/cd-diagram-regression-big-ds.png?raw=true)

### Regression tasks with small datasets

![Critical Difference Diagram for Classification tasks with bigger datasets](results/cd-diagram-regression-small-ds.png?raw=true)

### Classification tasks with small datasets

![Critical Difference Diagram for Regression tasks](results/cd-diagram-classification-small-ds.png?raw=true)

No critical difference diagram was created for classification tasks with big datasets since too few big classification datasets were included in the study.

## Installation
  1. Clone this repository
  2. Execute `conda env creae -f environment.yml -n ml-algorithms-comparison` in the repos to install the dependencies.
  3. Execute `conda activate ml-algorithms-comparison` to activate the newly created environment
  4. Execute `python main.py --help` in the repos to see usage information.
## Report
For more information, read [the full report](report.pdf?raw=true).
