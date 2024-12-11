![](UTA-DataScience-Logo.png)

# Metastatic Cancer Diagnosis

This project applies logistic regression to predict diagnoses using the "Metastatic Cancer Diagnosis" dataset from Kaggle, focusing on data cleaning, feature preparation, and performance evaluation.
## Overview

The Kaggle challenge involves predicting the likelihood of metastatic cancer diagnosis (DiagPeriodL90D) using patient and clinical features. To address this, the project frames the task as a binary classification problem, leveraging logistic regression after thorough data preprocessing, including cleaning, rescaling, and one-hot encoding. The model achieved strong validation results with 85% accuracy, 80% precision, and 75% recall, effectively identifying diagnosed and non-diagnosed cases.

## Summary of Workdone

Type: The dataset includes CSV files containing clinical and patient features as input. The target variable (DiagPeriodL90D) indicates whether a metastatic cancer diagnosis occurred within 90 days.

Size:

Training Set: 10,000 rows and 15 columns (features and target).
Test Set: 2,000 rows and 14 columns (features only, no target).
Instances:

Training Set: Comprises 70% of the data, used to train the model.
Validation Set: 15% of the data, used to tune and evaluate the model during training.
Test Set: The remaining 15%, reserved for final evaluation or Kaggle submission.

#### Preprocessing / Clean up

* Describe any manipulations you performed to the data.

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







