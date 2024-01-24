# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:49:07 2024

@author: HP
"""
# -------------Instructions:
# Import you data and perform basic data exploration phase
# Display general information about the dataset
# Create a pandas profiling reports to gain insights into the dataset
# Handle Missing and corrupted values
# Remove duplicates, if they exist
# Handle outliers, if they exist
# Encode categorical features
# Select your target variable and the features
# Split your dataset to training and test sets
# Based on your data exploration phase select a ML classification algorithm and train it on the training set
# Assess your model performance on the test set using relevant evaluation metrics
# Discuss with your cohort alternative ways to improve your model performance


import warnings
import pandas as pd
import seaborn as sns
import sweetviz as sv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# Importing the Dataset
data = pd.read_csv("C:/Users/HP/Downloads/African_crises_dataset (1).csv")
data1 = pd.read_csv("C:/Users/HP/Downloads/African_crises_dataset (1).csv")


# Exploratory Data Analysis
data.info()
data_head = data.head()
data_tail = data.tail()
data_descriptive_statistic = data.describe()
data_more_desc_statistic = data.describe(include = "all")
data_mode = data.mode()
data_distinct_count = data.nunique()
data_correlation_matrix = data.corr() 
data_null_count = data.isnull().sum()
data_total_null_count = data.isnull().sum().sum()
data_hist = data.hist(figsize = (15, 10), bins = 10)

# pandas profiling reports to gain insights into the dataset
profile_report = sv.analyze(data)
profile_report.show_html('profile_report.html')

data = data.drop("country_code", axis = 1)
data = data.drop("country", axis = 1)

# HANDLING CATEGORECAL VALUES
encoder = LabelEncoder()
data['banking_crisis'] = encoder.fit_transform(data['banking_crisis'])


# Further Data Preparation and Segregation
x = data.iloc[:, [0,1,3,7,8,9,10]]
y = data.banking_crisis

# Splitting the data into training and testing sets (default is 70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)


# Model Training
model = LogisticRegression(random_state = 0)
model.fit(x_train, y_train)

# Model Prediction
y_pred = model.predict(x_test)

# MODEL EVALUATION
acc = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test,y_pred)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
