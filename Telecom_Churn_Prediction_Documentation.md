---
author: Sanan Shahid
title: Telecom Customer Churn Prediction
---

# Telecom Customer Churn Prediction

## 1. Project Overview

This project is a machine learning application for predicting whether a
telecom customer is likely to churn.

The project contains two main parts:

1.  **Machine Learning / Data Analysis Notebook** --- used to load and
    inspect the telecom churn dataset, analyze the target and features,
    check data quality, and explore relationships between the target
    variable and the input features.
2.  **Streamlit Web Application** --- provides an interactive interface
    where a user enters customer information and receives a churn
    prediction, churn probability, and risk indicator.

The application uses saved model artifacts so that a trained model can
be used for prediction without retraining every time the Streamlit app
starts.

------------------------------------------------------------------------

# 2. Project Objectives

The main objectives are:

-   Analyze telecom customer churn data.
-   Understand the structure and quality of the dataset.
-   Use customer information as model input.
-   Predict the customer's churn class.
-   Display the predicted churn probability when the saved model
    supports probability prediction.
-   Provide an easy-to-use web interface.
-   Make the application visually interactive and responsive.
-   Apply the same saved scaler used during model development to new
    customer queries.
-   Keep the prediction input columns aligned with the columns used
    during training.

------------------------------------------------------------------------

# 3. Dataset

The project uses a telecom churn dataset loaded from:

``` python
pd.read_csv(r"C:\Users\SANAN\Documents\telecom_churn.csv")
```

The dataset contains **3,333 rows** and **11 columns**.

The target variable is:

``` text
Churn
```

The project uses the following 10 customer features as prediction
inputs:

  Feature             Description
  ------------------- -------------------------------------------------
  `AccountWeeks`      Number of weeks the customer has had an account
  `ContractRenewal`   Contract renewal indicator
  `DataPlan`          Data plan indicator
  `DataUsage`         Customer data usage
  `CustServCalls`     Number of customer service calls
  `DayMins`           Minutes used during the day
  `DayCalls`          Number of daytime calls
  `MonthlyCharge`     Monthly customer charge
  `OverageFee`        Overage fee
  `RoamMins`          Roaming minutes

The target is:

``` text
Churn
```

where the dataset contains binary values representing the churn class.

------------------------------------------------------------------------

# 4. Dataset Structure

The notebook shows the following data types:

-   `Churn`: integer
-   `AccountWeeks`: integer
-   `ContractRenewal`: integer
-   `DataPlan`: integer
-   `DataUsage`: float
-   `CustServCalls`: integer
-   `DayMins`: float
-   `DayCalls`: integer
-   `MonthlyCharge`: float
-   `OverageFee`: float
-   `RoamMins`: float

The dataset contains:

``` text
3333 entries
11 columns
5 float64 columns
6 int64 columns
```

The notebook also checks for missing values and duplicate rows.

The recorded results show:

``` text
Missing values: 0
Duplicate rows: 0
```

------------------------------------------------------------------------

# 5. Target Distribution

The notebook checks the distribution of the `Churn` target using:

``` python
df["Churn"].value_counts()
```

The recorded class counts are:

``` text
Churn = 0    2850
Churn = 1     483
```

This means the dataset is **class-imbalanced**, because the two classes
do not have equal numbers of examples.

The approximate distribution is:

-   Class 0: 85.5%
-   Class 1: 14.5%

This is important when evaluating a churn prediction model. Accuracy
alone may not completely describe how well the model handles the smaller
churn class.

------------------------------------------------------------------------

# 6. Exploratory Data Analysis

The notebook performs basic exploratory analysis using:

``` python
df.head()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

These operations are used to understand:

-   Dataset structure
-   Column data types
-   Number of records
-   Statistical summaries
-   Missing values
-   Duplicate records

The notebook also creates boxplots to examine the relationship between
`Churn` and the available features.

For numeric features, the notebook groups values into quartiles using:

``` python
pd.qcut(
    df[col],
    q=4,
    duplicates="drop"
)
```

and then creates boxplots against the `Churn` target.

------------------------------------------------------------------------

# 7. Machine Learning Workflow

The complete project workflow can be summarized as:

``` text
Telecom Dataset
      ↓
Data Loading
      ↓
Data Inspection
      ↓
Data Quality Checks
      ↓
Exploratory Data Analysis
      ↓
Feature / Target Separation
      ↓
Model Training
      ↓
Scaler Creation
      ↓
Model Saving
      ↓
Streamlit Application
      ↓
User Input
      ↓
Input DataFrame
      ↓
Training Column Alignment
      ↓
Saved Scaler
      ↓
Saved Model
      ↓
Prediction
      ↓
Churn Probability
      ↓
Risk Indicator
```

The exact machine learning estimator used for the final saved model is
not specified by the supplied notebook/app source, so this documentation
does not assume a particular algorithm.

------------------------------------------------------------------------

# 8. Saved Model Files

The Streamlit application loads three saved files:

``` text
churn_model.pkl
churn_scaler.pkl
churn_columns.pkl
```

## `churn_model.pkl`

Contains the trained machine learning model.

The app loads it using:

``` python
model = joblib.load("churn_model.pkl")
```

## `churn_scaler.pkl`

Contains the scaler fitted during model development.

The app loads it using:

``` python
scaler = joblib.load("churn_scaler.pkl")
```

The same scaler is applied to new user input before prediction.

## `churn_columns.pkl`

Contains the feature columns used by the trained model.

The app loads it using:

``` python
columns = joblib.load("churn_columns.pkl")
```

The saved column list is used to ensure that the new prediction query
has the expected columns and order.

------------------------------------------------------------------------

# 9. Streamlit Application

The web application is built with **Streamlit**.

The application provides a user-friendly interface for entering all 10
model features.

The input fields include:

``` text
Account Weeks
Contract Renewal
Data Plan
Data Usage
Customer Service Calls
Day Minutes
Day Calls
Monthly Charge
Overage Fee
Roaming Minutes
```

The user then clicks:

``` text
🔮 Predict Customer Churn
```

------------------------------------------------------------------------

# 10. Input Processing

When the user submits the form, the application creates a dictionary:

``` python
raw_input = {
    "AccountWeeks": account_weeks,
    "ContractRenewal": contract_renewal,
    "DataPlan": data_plan,
    "DataUsage": data_usage,
    "CustServCalls": cust_serv_calls,
    "DayMins": day_mins,
    "DayCalls": day_calls,
    "MonthlyCharge": monthly_charge,
    "OverageFee": overage_fee,
    "RoamMins": roam_mins
}
```

The dictionary is converted into a pandas DataFrame:

``` python
input_df = pd.DataFrame([raw_input])
```

------------------------------------------------------------------------

# 11. Training Column Matching

The application makes sure that the new query matches the saved training
columns.

It checks each expected column:

``` python
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0
```

It then applies the exact saved column order:

``` python
input_df = input_df[columns]
```

This step helps prevent errors caused by missing or incorrectly ordered
model features.

------------------------------------------------------------------------

# 12. Scaling

The new input is transformed using the same saved scaler:

``` python
scaled_input = scaler.transform(input_df)
```

Using the saved scaler is important because the model expects input in
the same numerical representation used during training.

------------------------------------------------------------------------

# 13. Prediction

The model predicts the customer's churn class using:

``` python
prediction = model.predict(scaled_input)[0]
```

The application interprets the prediction as:

``` text
1 → Customer is likely to churn
0 → Customer is unlikely to churn
```

------------------------------------------------------------------------

# 14. Churn Probability

If the saved model provides probability prediction through
`predict_proba`, the application calculates:

``` python
probability = model.predict_proba(scaled_input)[0]

churn_probability = probability[1] * 100
```

The probability is displayed as a percentage.

Example:

``` text
🎯 72.35%
```

means the model returned a 72.35% probability for the churn class.

------------------------------------------------------------------------

# 15. Risk Indicator

The application converts the probability into three display categories:

``` text
Below 30%  → LOW RISK
30%–69.99% → MEDIUM RISK
70%+       → HIGH RISK
```

These are **application display thresholds**, not additional machine
learning classes.

The risk indicator is shown with different visual styles:

``` text
🎉 LOW RISK
⚡ MEDIUM RISK
⚠️ HIGH RISK
```

------------------------------------------------------------------------

# 16. User Interface Features

The Streamlit application includes several interactive design features.

## Animated Header

The application has a gradient animated header containing:

``` text
📱 Telecom Customer Churn Prediction
AI-powered customer churn prediction system
```

## Interactive Input Cards

Each customer feature is displayed in a card-style section with an icon
and short description.

## Loading Animation

After the user clicks the prediction button, the application temporarily
displays:

``` text
🔄 Analyzing customer information...
AI model is processing the input...
```

## Animated Prediction Result

The result appears inside an animated result card.

## Churn Probability

The predicted churn probability is displayed as a percentage.

## Progress Bar

A visual progress bar represents the churn probability.

## Risk Indicator

The application displays:

``` text
LOW RISK
MEDIUM RISK
HIGH RISK
```

depending on the calculated probability.

## Success Animation

For a non-churn prediction, the application displays a success message
and Streamlit balloons.

## Warning Animation

High-risk results use a warning-style animated visual effect.

## Responsive Layout

The CSS includes responsive rules so the interface can adapt to smaller
screens.

------------------------------------------------------------------------

# 17. Technologies Used

## Python

Used as the main programming language.

## Pandas

Used for:

-   Loading data
-   DataFrame creation
-   Data inspection
-   Preparing prediction input

Example:

``` python
import pandas as pd
```

## NumPy

Used in the analysis notebook.

``` python
import numpy as np
```

## Matplotlib

Used for visualization in the notebook.

``` python
import matplotlib.pyplot as plt
```

## Seaborn

Used to create exploratory boxplots.

``` python
import seaborn as sns
```

## Scikit-learn

Used for the machine learning workflow and saved model/scaler artifacts.

## Joblib

Used to save and load trained model objects.

``` python
import joblib
```

## Streamlit

Used to create the interactive web application.

------------------------------------------------------------------------

# 18. Suggested Project Structure

The project can be organized as:

``` text
telecom-churn-prediction/
│
├── app.py
│
├── telecom_churn.csv
│
├── churn_model.pkl
├── churn_scaler.pkl
├── churn_columns.pkl
│
├── telecom_churn.ipynb
│
├── requirements.txt
│
└── README.md
```

------------------------------------------------------------------------

# 19. Requirements

A basic `requirements.txt` can contain:

``` text
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

The exact package versions can be pinned if deployment requires
reproducible environments.

------------------------------------------------------------------------

# 20. Running the Project Locally

## Step 1 --- Create or activate your Python environment

Use the Python environment where the required packages are installed.

## Step 2 --- Install dependencies

Run:

``` bash
pip install -r requirements.txt
```

## Step 3 --- Keep model files beside the app

Make sure these files are available in the application directory:

``` text
churn_model.pkl
churn_scaler.pkl
churn_columns.pkl
```

## Step 4 --- Run Streamlit

Run:

``` bash
streamlit run app.py
```

Streamlit will open the application in your browser.

------------------------------------------------------------------------

# 21. How to Use the Application

1.  Open the Streamlit application.
2.  Enter the customer's account information.
3.  Select whether the customer has renewed their contract.
4.  Select whether the customer has a data plan.
5.  Enter usage and billing information.
6.  Click **Predict Customer Churn**.
7.  Wait for the loading animation.
8.  View the predicted churn class.
9.  Check the churn probability.
10. Review the displayed risk indicator.

------------------------------------------------------------------------

# 22. Error Prevention

A major part of the application is maintaining consistency between
training and prediction.

The application loads:

``` python
columns = joblib.load("churn_columns.pkl")
```

and then uses:

``` python
input_df = input_df[columns]
```

This helps ensure that the model receives the expected feature names and
order.

The scaler is also loaded rather than recreated:

``` python
scaler = joblib.load("churn_scaler.pkl")
```

This prevents the application from fitting a new scaler to a single
customer query.

------------------------------------------------------------------------

# 23. Important Notes

### Model compatibility

The three `.pkl` files must correspond to the same trained model
pipeline/workflow.

### Feature compatibility

The input features in the Streamlit application must match the features
used to train the saved model.

### Probability availability

The churn probability is displayed when the loaded model provides:

``` python
predict_proba()
```

### Class imbalance

The dataset contains substantially more non-churn examples than churn
examples. Model evaluation should therefore consider metrics beyond
accuracy, especially metrics that describe performance on the churn
class.

### Risk thresholds

The 30% and 70% thresholds are UI rules used by this application to
categorize probability into low, medium, and high risk. They are not
learned automatically by the model.

------------------------------------------------------------------------

# 24. Project Learning Outcomes

Through this project, the following practical skills are demonstrated:

-   Loading a real-world dataset with Pandas
-   Inspecting DataFrame structure
-   Checking data types
-   Checking missing values
-   Checking duplicate records
-   Examining target distribution
-   Using descriptive statistics
-   Performing exploratory data visualization
-   Separating prediction features from the target
-   Preparing data for machine learning
-   Saving trained machine learning artifacts
-   Loading trained models with Joblib
-   Applying a saved scaler to new data
-   Maintaining feature-column consistency
-   Creating a machine learning prediction interface
-   Building a Streamlit application
-   Designing interactive input components
-   Displaying prediction probabilities
-   Creating risk indicators
-   Adding custom CSS and animations
-   Building a responsive machine learning web application

------------------------------------------------------------------------

# 25. Limitations

This application should be treated as a machine learning prediction tool
rather than a guarantee of future customer behavior.

The output depends on:

-   The quality of the training dataset
-   The trained model
-   The preprocessing/scaling process
-   The features supplied by the user
-   The model's ability to generalize to new customers

The application does not itself establish the reason a customer will
churn. It only presents the prediction produced by the saved model.

------------------------------------------------------------------------

# 26. Future Improvements

Possible future improvements include:

-   Add model evaluation metrics to the application.
-   Add confusion matrix visualization.
-   Add feature importance visualization when supported by the model.
-   Add model comparison.
-   Add customer prediction history.
-   Add CSV batch prediction.
-   Add downloadable prediction reports.
-   Add authentication.
-   Deploy the application online.
-   Add monitoring for model performance.
-   Add a dedicated explanation section for individual predictions.
-   Tune probability thresholds using validation data rather than fixed
    display thresholds.

------------------------------------------------------------------------

# 27. Conclusion

The Telecom Customer Churn Prediction project combines exploratory data
analysis, machine learning model usage, preprocessing consistency, and
an interactive Streamlit interface.

The final application allows a user to enter 10 telecom customer
features and receive a churn prediction together with a
probability-based risk display.

The project demonstrates the complete transition from a machine learning
dataset and analysis workflow to an interactive prediction application.
