# Soccer Match Outcome Prediction
![image for soccer analytics](images/img1.jpeg)

## Introduction
This project focuses on predicting the outcome of soccer matches using historical game and player data. The prediction model is built using logistic regression, and the data is obtained from the SportsData API.

## Data Retrieval
The project begins by fetching game and player data from the SportsData API using Python. The data is retrieved in two parts: game statistics (goals, possession, etc.) and player game stats (individual player performances). The data is collected over a specified date range.

```python
# Usage for retrieving game and player data
start_date = date(2023, 8, 12)
end_date = date(2023, 11, 30)
get_sportsdata_io_data(start_date, end_date)
```

## Team Data Retrieval
Additionally, team game stats are obtained to enrich the dataset. This includes team-specific statistics for each match.

```python
# Usage for retrieving team game data
start_date = '2023-08-12'
end_date = '2023-11-30'
get_sportsdata_io_data(start_date, end_date)
```

## Data Merging
The collected data is merged into a single dataset, `merged_data.csv`, using the 'GameId' column as the key.

```python
# Usage for merging and saving the data
import pandas as pd

# Load the CSV files
games_df = pd.read_csv('games_by_date_range.csv')
team_stats_df = pd.read_csv('team_game_stats.csv')

# Merge the tables on the 'GameId' column
merged_data = pd.merge(games_df, team_stats_df, on='GameId', how='inner')

# Save the merged data to a new CSV file
merged_data.to_csv('merged_data.csv', index=False)
```

## Data Preprocessing
The 'Winner' column is determined based on the scores, with labels such as 'HomeWin', 'AwayWin', and 'Draw'. The data is then prepared for the logistic regression model.

```python
# Usage for data preprocessing
# Replace the Winner column with encoded values
# Set the winner based on scores
# Drop unnecessary columns used for calculations
# Save the final preprocessed data
import pandas as pd

merged_data['Winner'] = 'Draw'  # Initialize all matches as draws
merged_data.loc[merged_data['AwayTeamScore'] > merged_data['HomeTeamScore'], 'Winner'] = 'AwayWin'  # Away win
merged_data.loc[merged_data['AwayTeamScore'] < merged_data['HomeTeamScore'], 'Winner'] = 'HomeWin'  # Home win
merged_data['Winner'] = merged_data['Winner'].apply(lambda x: 'HomeWin' if x == 'Draw' else x)
merged_data = merged_data.drop(['AwayTeamScore', 'HomeTeamScore'], axis=1)
merged_data.to_csv('preprocessed_data.csv', index=False)
```

## Model Training
The logistic regression model is trained on the preprocessed data. The relevant features are selected, and missing values are imputed.

```python
# Usage for model training
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the merged data
merged_data_df = pd.read_csv('preprocessed_data.csv')

# Feature selection
# Drop rows with missing 'Winner' values
# Separate features and target variable
# Handle missing values using SimpleImputer
# Encode the 'Winner' column
# Split the data into training and testing sets
# Initialize the logistic regression model
# Train the model
# Evaluate the model accuracy
# Display the coefficients for each feature
```

## Model Prediction
A function is created to predict the winner and their probability based on input teams.

```python
# Usage for model prediction
# Convert team names to 'AwayTeamId' and 'HomeTeamId'
# Create a DataFrame for the input teams
# Impute missing values
# Make predictions using the model
# Decode numeric labels back to team names
# Sort predictions by probabilities
# Print the predicted winner and probability
```

## Example Usage
An example usage is provided where users input the home and away teams, and the model returns the predicted winner with the probability.

```python
# Example usage for model prediction
home_team_input = 'Arsenal FC'
away_team_input = 'Manchester United FC'

try:
    winner, probability = predict_winner_probability(home_team_input, away_team_input, model, label_encoder, imputer)
    print(f"Predicted Winner: {winner} with Probability: {probability:.2%}")
except ValueError as e:
    print(f"Error: {e}")
```

Feel free to use and adapt this code for your own soccer match outcome prediction project!