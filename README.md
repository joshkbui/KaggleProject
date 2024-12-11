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

Handling Missing Values:

Identified columns with missing values.
For numerical features, missing values were replaced with the column's median.
For categorical features, missing values were replaced with the most frequent value (mode).
Removing Irrelevant Columns:

Dropped columns that were unnecessary for prediction, such as unique identifiers like patient_id.
Rescaling Numerical Features:

Applied Min-Max Scaling to transform numerical features to a uniform range of [0, 1], ensuring consistency across features and improving model performance.
Encoding Categorical Variables:

Applied one-hot encoding to categorical features, converting them into binary columns for compatibility with machine learning models.
Data Splitting:

Split the data into training (70%), validation (15%), and test (15%) sets to evaluate model performance.

#### Data Visualization

These visualizations suggest that age, income, and to a lesser extent, gender, could be important predictors for the likelihood of diagnosis (DiagPeriodL90D). The target class imbalance indicates the need for strategies to address potential bias in the model. ![image](https://github.com/user-attachments/assets/f20e2249-264d-46ed-83d5-025ac0637137)![image](https://github.com/user-attachments/assets/7f2f3e20-8a1c-4c61-84b9-9dee093baac8)![image](https://github.com/user-attachments/assets/2e49e012-cd4d-4a5f-8901-16a00d0bff14)






### Problem Formulation

Input / Output
Input: The dataset includes clinical and patient features derived from a CSV file. After preprocessing, features consist of both scaled numerical data and one-hot encoded categorical variables.
Output: The target variable, DiagPeriodL90D, is a binary classification indicating whether a metastatic cancer diagnosis occurred within 90 days (1 = Yes, 0 = No).
Models
Logistic Regression:
This was chosen as the primary model due to its simplicity, interpretability, and effectiveness for binary classification problems.
Logistic regression provides probabilities for class predictions, which can be further evaluated with metrics like ROC-AUC.
Why Used:
Computational efficiency.
Baseline comparison for future, more complex models.
Loss, Optimizer, and Hyperparameters
Loss: Binary cross-entropy loss was implicitly minimized as logistic regression is designed for binary classification.
Optimizer: Stochastic gradient descent (SGD) is used internally by the logistic regression implementation to minimize the loss.
Hyperparameters:
max_iter = 1000: Ensured the algorithm converged by allowing up to 1000 iterations.
random_state = 42: Ensured reproducibility of results.
Default settings for learning rate and other parameters were used to focus on achieving a functional baseline result.


### Training


Training
How You Trained
Software: Python with Scikit-Learn for logistic regression.
Hardware: Standard laptop/PC with a modern CPU and 8-16 GB of RAM. No GPU was needed.
Training Time
Training took less than 30 seconds due to the simplicity of logistic regression and the dataset size.
Training Curves
Logistic regression doesn’t involve epochs, so no loss vs. epoch curves were generated. Performance was evaluated using metrics like accuracy, precision, and ROC-AUC on the validation set.
Stopping Criteria
Training stopped after convergence or a maximum of 1000 iterations.
Difficulties
Class imbalance: Resolved using class weighting.
Data preparation: Addressed through thorough preprocessing, including scaling and encoding features.

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







