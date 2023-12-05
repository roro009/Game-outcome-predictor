#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

features = [
    'PTS_diff', 'FG_PCT_diff', 'FT_PCT_diff', 'FG3_PCT_diff', 
    'AST_diff', 'REB_diff', 'SEASON'
    ]
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("C:/Users/rohan/OneDrive/Desktop/INTRO TO INFORMATICS/Final_project/bg.avif");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Function to load and preprocess data
def load_data():
    file_path = 'C:/Users/rohan/OneDrive/Desktop/INTRO TO INFORMATICS/LAB/Data_for_project/archive/games.csv'  # Update with the correct path
    games_df = pd.read_csv(file_path)
    # Data Cleaning and Preprocessing

    # Dropping irrelevant or redundant columns
    games_cleaned_df = games_df.drop(columns=['GAME_DATE_EST', 'GAME_STATUS_TEXT', 'Unnamed: 21', 'TEAM_ID'])

    # Checking for missing values
    missing_values = games_cleaned_df.isnull().sum()

    # Converting data types where necessary
    games_cleaned_df = games_cleaned_df.convert_dtypes()

    # Feature Engineering: Creating new features that might be useful for the model
    # Calculating the difference in points, field goal percentage, free throw percentage, etc.
    games_cleaned_df['PTS_diff'] = games_cleaned_df['PTS_home'] - games_cleaned_df['PTS_away']
    games_cleaned_df['FG_PCT_diff'] = games_cleaned_df['FG_PCT_home'] - games_cleaned_df['FG_PCT_away']
    games_cleaned_df['FT_PCT_diff'] = games_cleaned_df['FT_PCT_home'] - games_cleaned_df['FT_PCT_away']
    games_cleaned_df['FG3_PCT_diff'] = games_cleaned_df['FG3_PCT_home'] - games_cleaned_df['FG3_PCT_away']
    games_cleaned_df['AST_diff'] = games_cleaned_df['AST_home'] - games_cleaned_df['AST_away']
    games_cleaned_df['REB_diff'] = games_cleaned_df['REB_home'] - games_cleaned_df['REB_away']

    # Handling missing values
    # Using SimpleImputer to fill missing values with the mean of each column

    imputer = SimpleImputer(strategy='mean')
    games_imputed_df = pd.DataFrame(imputer.fit_transform(games_cleaned_df), columns=games_cleaned_df.columns)
    games_imputed_df['HOME_TEAM_WINS'] = games_imputed_df['HOME_TEAM_WINS'].round().astype(int)

    y = games_imputed_df['HOME_TEAM_WINS']
    # Feature Selection: Selecting potential features for the model
    # Here, we're excluding identifiers like GAME_ID, TEAM_IDs, and focusing on game stats
   
    # Calculating home averages
    home_stats = games_imputed_df.groupby('TEAM_ID_home').mean()[[
    'PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home'
    ]]

    # Calculating away averages
    away_stats = games_imputed_df.groupby('TEAM_ID_away').mean()[[
    'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away'
    ]]

    # Combining these averages to get overall averages for each team
    combined_stats = (home_stats.add(away_stats, fill_value=0) / 2).reset_index()
    combined_stats.rename(columns={'TEAM_ID_home': 'TEAM_ID'}, inplace=True)


    return games_imputed_df, combined_stats

# Function to make predictions
def make_prediction(model, team_1_id, team_2_id, combined_stats, most_recent_season):
    # Retrieve average stats for each team
    team_1_stats = combined_stats[combined_stats['index'] == team_1_id].iloc[0]
    team_2_stats = combined_stats[combined_stats['index'] == team_2_id].iloc[0]

    # Prepare the input features
    input_features = pd.DataFrame({
        'PTS_diff': team_1_stats['PTS_home'] - team_2_stats['PTS_away'],
        'FG_PCT_diff': team_1_stats['FG_PCT_home'] - team_2_stats['FG_PCT_away'],
        'FT_PCT_diff': team_1_stats['FT_PCT_home'] - team_2_stats['FT_PCT_away'],
        'FG3_PCT_diff': team_1_stats['FG3_PCT_home'] - team_2_stats['FG3_PCT_away'],
        'AST_diff': team_1_stats['AST_home'] - team_2_stats['AST_away'],
        'REB_diff': team_1_stats['REB_home'] - team_2_stats['REB_away'],
        'SEASON': most_recent_season
    }, index=[0])

    # Add any additional required features with default values
    for feature in model.feature_names_in_:
        if feature not in input_features:
            input_features[feature] = 0

    # Order columns as expected by the model
    input_features = input_features[model.feature_names_in_]

    # Make the prediction
    probability_team_1_wins = model.predict_proba(input_features)[:, 1][0]
    probability_team_2_wins = 1 - probability_team_1_wins

    return probability_team_1_wins, probability_team_2_wins

# Streamlit app layout
def main():
    add_bg_from_url() 
    st.title('Game Outcome Prediction')
    
    # Load and preprocess data
    games_imputed_df, combined_stats = load_data()

    # Train model (or load a pre-trained model)
    # Ensuring the target variable 'HOME_TEAM_WINS' is categorical after imputation
    games_imputed_df['HOME_TEAM_WINS'] = games_imputed_df['HOME_TEAM_WINS'].round().astype(int)

    # Preparing the data for modeling again
    X = games_imputed_df[features]
    y = games_imputed_df['HOME_TEAM_WINS']
    
    # Splitting the dataset into training and testing sets again
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Training the RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicting and Evaluating the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # User input for team selection
    team_1_id = st.selectbox('Select Team 1', combined_stats['index'])
    team_2_id = st.selectbox('Select Team 2', combined_stats['index'])
    
    if st.button('Predict Outcome'):
        most_recent_season = games_imputed_df['SEASON'].max()
        prob_team_1, prob_team_2 = make_prediction(model, team_1_id, team_2_id, combined_stats, most_recent_season)
        st.write(f'Probability of Team 1 Winning: {prob_team_1 * 100:.2f}%')
        st.write(f'Probability of Team 2 Winning: {prob_team_2 * 100:.2f}%')

if __name__ == '__main__':
    main()


# In[ ]:




