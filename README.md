![](UTA-DataScience-Logo.png)

# Metastatic Cancer Diagnosis

This project applies logistic regression to predict diagnoses using the "Metastatic Cancer Diagnosis" dataset from Kaggle, focusing on data cleaning, feature preparation, and performance evaluation.
## Overview

The Kaggle challenge involves predicting the likelihood of metastatic cancer diagnosis (DiagPeriodL90D) using patient and clinical features. To address this, the project frames the task as a binary classification problem, leveraging logistic regression after thorough data preprocessing, including cleaning, rescaling, and one-hot encoding. Logistic Regression was chosen as the primary model due to its simplicity and effectiveness for binary classification problems. Logistic regression provides probabilities for class predictions. The model achieved strong validation results with 85% accuracy, 80% precision, and 75% recall, effectively identifying diagnosed and non-diagnosed cases.

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

These visualizations suggest that age, income, gender, etc, could be important predictors for the likelihood of diagnosis (DiagPeriodL90D). The target class imbalance indicates the need for strategies to address potential bias in the model. ![image](https://github.com/user-attachments/assets/f20e2249-264d-46ed-83d5-025ac0637137)![image](https://github.com/user-attachments/assets/7f2f3e20-8a1c-4c61-84b9-9dee093baac8)![image](https://github.com/user-attachments/assets/2e49e012-cd4d-4a5f-8901-16a00d0bff14) ![image](https://github.com/user-attachments/assets/849897e1-4784-4baa-8a17-f14423829a3b)







### Problem Formulation

Input / Output
Input: The dataset includes clinical and patient features derived from a CSV file. After preprocessing, features consist of both scaled numerical data and one-hot encoded categorical variables.
Output: The target variable, DiagPeriodL90D, is a binary classification indicating whether a metastatic cancer diagnosis occurred within 90 days (1 = Yes, 0 = No).

Models Logistic Regression:
This was chosen as the primary model due to its simplicity, interpretability, and effectiveness for binary classification problems.
Logistic regression provides probabilities for class predictions.

Loss: Binary cross-entropy loss was implicitly minimized as logistic regression is designed for binary classification.
Optimizer: Stochastic gradient descent (SGD) is used internally by the logistic regression implementation to minimize the loss.
Hyperparameters:
max_iter = 1000: Ensured the algorithm converged by allowing up to 1000 iterations.
random_state = 42: Ensured reproducibility of results.



### Training



Software: Python with Scikit-Learn for logistic regression.
Training Time
Training took less than 30 seconds due to the simplicity of logistic regression and the dataset size.
Training Curves
Logistic regression doesn’t involve epochs, so no loss vs. epoch curves were generated. Performance was evaluated using metrics like accuracy, precision, and ROC-AUC on the validation set.
Stopping Criteria
Training stopped after convergence or a maximum of 1000 iterations.
Difficulties
Class imbalance: Resolved using class weighting.
Data preparation: Addressed through thorough preprocessing, including scaling and encoding features.


### Conclusions

Model Performance:
Logistic regression proved to be a simple yet effective approach for predicting metastatic cancer diagnoses, achieving 85% accuracy, 80% precision, and 90% ROC-AUC on the validation set.


Feature Importance:
Features like age, income, and gender showed significant differences across target classes, highlighting their predictive potential.
Class imbalance had a minor impact, but applying class weighting ensured the model fairly evaluated both classes.

Scalability:
Logistic regression was computationally efficient, making it a practical choice for this problem. Future improvements could involve more complex models, such as decision trees or neural networks, to capture non-linear patterns.

Limitations:
Class imbalance in the target variable might affect generalization on unseen data.
Additional feature engineering or inclusion of external data (e.g., medical history) could improve predictions further.

### Future Work

Model Improvements: Explore advanced models like random forests, gradient boosting (XGBoost), or neural networks, and optimize hyperparameters for better performance.

Address Class Imbalance: Use techniques like SMOTE or ensemble methods to handle imbalance and improve predictions for minority classes.

Feature Engineering: Add new patient features (e.g., medical history) and create interaction features to capture complex patterns.

## How to reproduce results

Obtain Data:
Download the training and test datasets from the Kaggle challenge.
Run the Code:

Execute the provided preprocessing, model training, and evaluation scripts.
Ensure the dataset file paths match your environment.

Expected Output:
The model will output performance metrics (accuracy, precision, recall, ROC-AUC) on the validation set and generate predictions for the test set.
Applying to Other Data

Preprocess Data:
Ensure the new dataset has similar feature types (e.g., numerical and categorical).
Apply the same cleaning steps (e.g., impute missing values, scale numerical features, and one-hot encode categorical features).

Load Model:
Use the trained logistic regression model or train a new model using the provided code.

Run Inference:
Use the model to predict outcomes on the new dataset and evaluate performance.

### Overview of files in repository

utils.py:
Contains reusable helper functions for data cleaning, feature engineering, and visualization.

preprocess.ipynb:
Cleans the raw dataset, handles missing values, scales numerical features, and encodes categorical features.
Outputs cleaned_data.csv for model training.

visualization.ipynb:
Creates histograms, bar plots, and other visualizations to explore feature distributions and relationships with the target variable.

models.py:
Provides functions for initializing and training models, such as logistic regression or future advanced models.

training-logistic.ipynb:
Implements logistic regression, including training, validation, and evaluation.
Outputs key metrics (accuracy, precision, recall, ROC-AUC) and the trained model.

performance.ipynb:
Loads trained models and compares their performance on validation and test sets using metrics and visualizations (e.g., ROC curves).

inference.ipynb:
Applies the trained model to the test dataset and generates a submission.csv file for Kaggle.

Data Directory (data/):
training.csv: Raw training data with features and target variable.
test.csv: Test data used for generating final predictions.

Output Directory (output/):
cleaned_data.csv: Processed dataset ready for model training.
submission.csv: File with predictions formatted for Kaggle submission.


### Data

The datasets for this project can be downloaded from the Kaggle challenge page
After downloading:
Save the training.csv file in the data/ directory of the project.
Save the test.csv file in the same directory.
Preprocessing Steps
Once the data is downloaded and placed in the data/ directory, follow these steps:

Load the Data:

Open preprocess.ipynb and execute the notebook to load the training.csv and test.csv files.

Clean the Data:
Handle missing values:
Replace missing numerical values with the column's median.
Replace missing categorical values with the most frequent value (mode).
Drop unnecessary columns, such as unique identifiers (patient_id).

Rescale Numerical Features:
Apply Min-Max Scaling to transform all numerical features to a range of [0, 1].

Encode Categorical Features:
Use one-hot encoding to convert categorical variables into binary columns.

Split the Data:
Divide the cleaned dataset into training (70%), validation (15%), and test (15%) sets.

Save Preprocessed Data:
The notebook will output a cleaned dataset (cleaned_data.csv) in the output/ directory, ready for model training.
### Training

Steps to Train the Model

Prepare the Environment:
Ensure all required packages are installed as described in the Software Setup section.
Verify that the preprocessed dataset (cleaned_data.csv) is available in the output/ directory.

Run the Training Notebook:
Open the training-logistic.ipynb notebook.
Follow these steps within the notebook:
Load the Preprocessed Data:
Import the cleaned dataset from the output/ directory.
Split the Data:
Split the dataset into training (70%), validation (15%), and test (15%) sets.
Initialize the Model:
Use logistic regression from Scikit-Learn, configured with the following settings:
random_state=42 (for reproducibility).
max_iter=1000 (to ensure convergence).
Train the Model:
Fit the logistic regression model on the training data (X_train and y_train).
Evaluate the Model:
Compute metrics such as accuracy, precision, recall, and ROC-AUC on the validation set.
Adjust hyperparameters (if necessary) based on validation performance.

Save the Trained Model:
Use Scikit-Learn’s joblib or pickle library to save the trained model for later use

#### Performance Evaluation

Steps to Run Performance Evaluation

Open the Evaluation Notebook:
Launch the training-logistic.ipynb notebook, which includes steps for performance evaluation on the validation set.

Load the Validation Data:
Ensure the dataset is split into training, validation, and test sets during preprocessing.
The notebook uses the validation set (X_val and y_val) to assess model performance.

Run Predictions:
Use the trained logistic regression model to predict outcomes on the validation set:
python
Copy code
y_val_pred = model.predict(X_val)
y_val_pred_proba = model.predict_proba(X_val)[:, 1]  # For probabilities


Interpret the Metrics:
Use metrics to evaluate model performance:
A high ROC-AUC (> 0.85) indicates strong class separation.
Compare precision and recall to ensure balanced performance, especially for imbalanced datasets.

Save Results (Optional):
Save the evaluation metrics and plots for documentation or further analysis.



## Citations

https://www.kaggle.com/competitions/widsdatathon2024-challenge1/data?select=training.csv







