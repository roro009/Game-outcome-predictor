# Game-outcome-predictor
## Abstract
The primary goal of this project was to develop a predictive model to forecast the outcomes of NBA games. Using historical data about games, teams, and player performances, we constructed a model to predict whether the home team would win a given game. The project involved data preprocessing, exploratory data analysis (EDA), model training, and evaluation, culminating in the deployment of a Streamlit web app. The final outcome was a RandomForestClassifier model with good accuracy, capable of predicting game outcomes and providing insights into team performance trends.

## Data Description
The data utilized in this project comprised various aspects of NBA games spanning over two decades, including game dates, team IDs, season information, and detailed performance metrics like points scored, field goal percentages, assists, and rebounds. The data initially contained missing values and some irrelevant columns, which were addressed through preprocessing steps. Columns such as 'Unnamed: 21' were removed, and missing values in critical columns were dropped to ensure data quality.

## Algorithm Description
The core of the web app is driven by a RandomForestClassifier, a robust ensemble learning method used for classification tasks. This model was selected for its ability to handle a large number of input features and its robustness against overfitting. The model predicts the likelihood of the home team winning a game based on historical game statistics. It was trained on a set of features including seasonal data and both home and away team statistics (like points scored, field goal percentages, etc.), and evaluated for accuracy.

##Tools Used
Python: The main programming language used for data manipulation, model building, and web app development.

Pandas: A Python library for data manipulation and analysis; used here for data loading, cleaning, and preprocessing.

Matplotlib & Seaborn: Python libraries for data visualization; used for creating histograms and other plots to explore data distributions and relationships.

Scikit-learn: A machine learning library in Python, employed for model training, prediction, and evaluation. It provided the RandomForestClassifier algorithm and functions for train-test split and cross-validation.

Streamlit: An open-source app framework for Machine Learning and Data Science teams. In this project, Streamlit was used to build and deploy the interactive web application that hosts the model.

NumPy: Used for numerical operations, especially in the context of handling model predictions and cross-validation scores.
