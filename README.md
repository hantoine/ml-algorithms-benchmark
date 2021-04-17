# Machine Learning Algorithms Benchmark
This repository contains the source code used to compare the performance of different classification and regression machine learning algorithms on tasks from the  [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). The datasets used are all relatively small structured datasets.

We used automatic hyperparameter tuning to ensure that the same effort is invested in the tuning of each algorithm on each dataset. We used [Tree-structured Parzen estimators](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) to perform automatic hyperparameter tuning. Basic feature engineering was performed for each dataset.

You can find more information in [the project report](report.pdf?raw=true).

## Results

### Classification

#### Scores on Test Sets
| Dataset             | Metric|   AB |   ANN |   BNN |   DT |   GB |   KNN |   LR |   RF |   SVM |
|:--------------------|:------|:----:|:-----:|:-----:|:----:|:----:|:-----:|:----:|:----:|:-----:|
| Adult               | f1    | 0.7  |  0.68 |  0.67 | 0.67 | 0.7  |  0.62 | 0.67 | **0.71** |  0.66 |
| BreastCancer        | f1    | 0.93 |  **0.95** |  0.94 | 0.91 | 0.94 |  0.92 | 0.94 | **0.95** |  **0.95** |
| DefaultCreditCard   | f1    | 0.46 |  0.53 |  0.47 | 0.52 | 0.47 |  0.44 | 0.49 | **0.54** |  0.53 |
| Retinopathy         | acc   | 0.71 |  0.74 |  0.71 | 0.67 | 0.72 |  0.66 | 0.74 | 0.67 |  **0.77** |
| SeismicBumps        | f1    |  F   |  0.25 |  0.28 | 0.27 | 0.1  |  0.2  | **0.32** | 0.28 |  0.3  |
| StatlogAustralian   | acc   | 0.83 |  0.82 |  0.83 | 0.83 | 0.83 |  **0.85** | 0.83 | 0.84 |  0.83 |
| StatlogGerman       | f1    | 0.59 |  **0.66** |  0.57 | 0.54 | 0.52 |  0.48 | 0.64 | 0.59 |  0.63 |
| SteelPlatesFaults   | f1    | 0.61 |  0.71 |  0.75 | 0.72 | **0.82** |  0.76 | 0.68 | 0.8  |  0.77 |
| ThoraricSurgery     | f1    |  F   |  0.27 |  0.3  | 0.26 | 0.08 |  0.16 | **0.36** | 0.24 |  0.35 |
| Yeast               | f1    | 0.38 |  0.55 |  0.52 | 0.35 | 0.46 |  0.58 | 0.57 | **0.6**  |  0.56 |
#### Critical Diagrams
##### All Classification Tasks
![Critical Difference Diagram for Classification tasks](results/cd-diagram-classification.png?raw=true)
##### Classification Tasks with Very Small Datasets
![Critical Difference Diagram for Regression tasks](results/cd-diagram-classification-small-ds.png?raw=true)

No critical difference diagram was created for classification tasks with big datasets since too few big classification datasets were included in the study.

### Regression
#### Ranks on Test Sets
  Dataset                                  |   AB | ANN   |   BNN |   DT | GP   |   GB |   KNN |   LR |   RF | SVM   |
|:-----------------------------------------|:----:|:-----:|:-----:|:----:|:----:|:----:|:-----:|:----:|:----:|:-----:|
| BikeSharing                              |  9   | 5     |   2   |  4   | 6    |  3   |   8   | 10   |  **1**   | 7     |
| CommunitiesAndCrime                      |  8   | 7     |   4   |  9   | 10   |  2   |   6   |  3   |  5   | **1**     |
| ConcreteCompressiveStrength              |  8   | 6     |   3   |  7   | 4    |  **1**   |   9   | 10   |  2   | 5     |
| FacebookComment                          |  3   | 7     |   **1**   | 10   | 6    |  2   |   8   |  4   |  9   | 5     |
| FacebookLikes                            |  9   | 10    |   **1**   |  8   | 7    |  3   |   4   |  5   |  6   | 2     |
| FacebookShare                            | 10   | 8     |   7   |  9   | 5    |  **1**   |   4   |  3   |  2   | 6     |
| MerckMolecularActivity                   |  9   | 6     |   3   |  8   | F    |  **1**   |   7   |  5   |  2   | 4     |
| ParkinsonMultipleSoundRecording          |  3   | 10    |   9   |  7   | 2    |  6   |   **1**   |  5   |  8   | 4     |
| QsarAquaticToxicity                      |  8   | 4     |   3   | 10   | 5    |  7   |   6   |  9   |  **1**   | 2     |
| RedWineQuality                           |  8   | 7     |   6   |  9   | 4    |  3   |   2   | 10   |  **1**   | 5     |
| SGEMMGPUKernelPerformances               |  6   | F     |   2   |  3   | F    |  5   |   4   |  7   |  **1**   | F     |
| StudentMathPerformance                   |  3   | 8     |   5   |  4   | 10   |  2   |   9   |  7   |  **1**   | 6     |
| StudentMathPerformanceNoPrevGrades       |  7   | 5     |   8   |  4   | 10   |  **1**   |   9   |  6   |  2   | 3     |
| StudentPortuguesePerformance             |  4   | 9     |   6   |  7   | 10   |  5   |   8   |  3   |  2   | **1**     |
| StudentPortuguesePerformanceNoPrevGrades |  7   | 10    |   3   |  8   | 9    |  **1**   |   6   |  4   |  5   | 2     |
| WhiteWineQuality                         |  8   | 7     |   6   | 10   | 5    |  2   |   3   |  9   |  **1**   | 4     |
| Average                                  |  6.9 | 7.3   |   4.3 |  7.3 | 6.6  |  **2.8** |   5.9 |  6.2 |  3.1 | 3.8   |
| Average Big Dataset                      |  8   | 5.5   |   2.3 |  5   | 6.0  |  3   |   6.3 |  7.3 |  **1.3** | 5.5   |
| Average Small Dataset                    |  6.6 | 7.5   |   4.8 |  7.8 | 6.7  |  **2.8** |   5.8 |  6   |  3.5 | 3.5   |
#### Critical Diagrams
##### All Regression Tasks
![Critical Difference Diagram for Regression tasks](results/cd-diagram-regression.png?raw=true)
##### Regression Tasks with Relatively Big Datasets
![Critical Difference Diagram for Classification tasks with bigger datasets](results/cd-diagram-regression-big-ds.png?raw=true)
##### Regression Tasks with Very Small Datasets
![Critical Difference Diagram for Classification tasks with bigger datasets](results/cd-diagram-regression-small-ds.png?raw=true)

## Installation
  1. Clone this repository
  2. Execute `conda env creae -f environment.yml -n ml-algorithms-comparison` to install dependencies.
  3. Execute `conda activate ml-algorithms-comparison` to activate the newly created environment
  4. Execute `python main.py --help` to see usage information.
