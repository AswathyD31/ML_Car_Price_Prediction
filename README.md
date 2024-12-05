# Car Price Prediction

This project aims to predict car prices based on various features using machine learning models. The dataset contains information about car attributes such as make, model, horsepower, engine size, and more. The goal is to build regression models to accurately predict car prices and perform feature importance analysis to understand which variables most significantly affect the price.

# Table of Contents

* Introduction 
* Data Description
* Setup
* Usage
* Models Implemented
* Evaluation Metrics
* Feature Importance
* Hyperparameter Tuning
* Contributors

# Introduction

This repository contains a car price prediction model using various regression techniques. The model aims to help businesses understand how different features affect car prices and how they can adjust pricing strategies for different markets.

## The following steps are performed in the project:

* **Data Preprocessing:** Outlier detection, skewness treatment, encoding categorical features, and scaling of features.

* **Model Implementation:**  Five regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regressor) are used to predict car prices.
  
* **Model Evaluation:**  The models are compared using R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).

* **Feature Importance:**  A feature importance analysis is performed to identify the most significant variables affecting car prices.
  
* **Hyperparameter Tuning:**  The best models are optimized using GridSearchCV and RandomizedSearchCV.
  
# Data Description
The dataset used for this project is the "Car Price Prediction" dataset. It contains various features about the cars, including:

* **Make, Model, Year:**  Categorical features describing the car.
  
* **Horsepower, Engine Size, Curb Weight:**  Numerical features.
* **Price:**  The target variable (car price).
  
## Setup
### Prerequisites
Before running the code, make sure you have the following libraries installed:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* scipy
  
The script will perform the following:

* Load and clean the dataset.
* Handle missing values, outliers, and skewness.
* Encode categorical variables and scale numerical features.
* Train and evaluate multiple regression models.
* Display evaluation metrics for each model.
* Perform feature importance analysis.
* Apply hyperparameter tuning to improve model performance.
  
# Models Implemented
The following regression models are used in this project:

* **Linear Regression:**  A simple linear model to predict car prices based on the input features.
  
* **Decision Tree Regressor:**  A tree-based model that splits the data into branches based on features to predict the target.
* Random Forest Regressor: An ensemble of decision trees, which aggregates predictions from multiple decision trees for better accuracy.
* Gradient Boosting Regressor: An ensemble model that builds trees sequentially, with each tree improving the prediction of the previous one.
* Support Vector Regressor (SVR): A model that tries to find the best hyperplane that minimizes the error for regression tasks.
  
# Evaluation Metrics
The models are evaluated based on the following metrics:

* **R-squared (R²):** The proportion of variance explained by the model.
  
* **Mean Squared Error (MSE):** The average squared difference between actual and predicted values.
* **Mean Absolute Error (MAE):** The average of the absolute errors between the predicted and actual values.
  
# Feature Importance
**The Random Forest Regressor**  model is used to analyze the importance of features. 

# Hyperparameter Tuning
**GridSearchCV** and **RandomizedSearchCV** are used to optimize the hyperparameters of the Random Forest Regressor and Gradient Boosting Regressor models. This helps in improving model performance by selecting the best parameters.
