### Overview

Financial institutions struggle with assessing loan applicants' creditworthiness, impacting their ability to manage risk and minimize defaults. Traditional credit evaluation methods may not capture the complexities of financial behaviour, leading to potential losses. Leveraging machine learning can enhance loan approval processes and decision-making.

### Challenge

Develop a predictive model to assess loan default risk based on applicant data. The goal is to help financial institutions improve lending decisions, reduce defaults, and enhance portfolio performance.

### Objectives

1. **Data Exploration and Preprocessing**:
   - Analyse the dataset to identify key features affecting loan default.
   - Clean and preprocess data for modelling.

2. **Model Development**:
   - Build and train a predictive model using algorithms like Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting Machines.

3. **Evaluation**:
   - Assess model performance with metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.
   - Ensure fairness and address biases, especially regarding income and employment.

4. **Actionable Insights**:
   - Identify factors impacting loan default risk.
   - Recommend how to integrate the model into loan approval processes.

5. **Documentation and Presentation**:
   - Document the process, model selection, and evaluation.
   - Prepare a presentation to communicate the model's performance and insights.

### Submission Requirements

- Well-documented code repository with run instructions.
- Detailed report or presentation summarizing approach, results, and insights.
- Demo or visualisation of model predictions and their impact.

### Evaluation Criteria

- Model accuracy and performance.
- Quality of data preprocessing and feature engineering.
- Fairness and bias in predictions.
- Clarity and usefulness of insights.
- Overall presentation and documentation.

### Dataset

- Kaggle: [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)

## Project Structure
Folder and File Descriptions:
### artifacts/: This folder contains the data/ and models/ subdirectories. It is automatically generated when the app runs and holds the trained models and preprocessed data.

data/: Contains the train.csv, test.csv, and the preprocessor.pkl file used for data transformation.
models/: Holds the trained models and their feature importance in .csv format.

### notebook/: Contains the Jupyter notebook used for experimentation and the Kaggle dataset.

### src/: The source folder containing the main codebase:

components/: Holds modules for data ingestion, transformation, and model training.
pipeline/: Contains the scripts for the prediction and training pipelines.

#### app.py: Main application script to run the FastAPI web app.

#### requirements.txt: Lists the required dependencies for the project.

#### logger.py: Handles logging for the project.

#### utils.py: Contains utility functions used across the project.

## Note: The artifacts/ folder is generated dynamically when you run the app using the command:
        uvicorn app:app --reload
