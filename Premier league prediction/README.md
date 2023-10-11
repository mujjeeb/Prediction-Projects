
# Premier League Football Match Prediction

![Premier League Logo](prediction.jpeg)

This project is a machine learning-based football match prediction system that aims to predict the outcomes of Premier League matches. The project is divided into two main parts: data scraping and match prediction.

## Table of Contents

- [Overview](#overview)
- [Data Scraping](#data-scraping)
- [Match Prediction](#match-prediction)
- [Dependencies](#dependencies)

## Overview

This project takes inspiration from DataQuest's "Predict Football Match Winners With Machine Learning And Python" project. The goal is to scrape data from the Premier League statistics website, clean and preprocess the data, and use machine learning models to predict the results of future matches.

## Data Scraping

In the first part of the project, we scrape data from "https://fbref.com/en/comps/9/Premier-League-Stats" using Python's requests and Beautiful Soup libraries. The script responsible for this is `premier_league_data_scrape.py`. The scraped data includes various statistics related to Premier League matches, which are essential for building our predictive models.

The scraped data is then cleaned and organized using the pandas library. The cleaned data is saved to a CSV file named `matches.csv`. This file serves as the dataset for our prediction models.

## Match Prediction

The second part of the project, found in the Jupyter Notebook file `premier_league_prediction.ipynb`, focuses on using machine learning models to predict future match outcomes. We utilize scikit-learn, a popular machine learning library in Python, to build two classification models:

1. **RandomForestClassifier**: This model leverages the power of random forests to make predictions based on the features extracted from our dataset.

2. **GradientBoostingClassifier**: We also employ gradient boosting, a powerful ensemble technique, to create a second prediction model.

In the Jupyter Notebook, you will find step-by-step explanations of the prediction process, feature selection, model training, and evaluation. The notebook is well-documented and designed to be interactive.

## Dependencies

To run this project, you will need the following dependencies:

- JupyterLab
- Python 3.3+
- Python packages:
  - pandas
  - requests
  - BeautifulSoup
  - scikit-learn

Make sure to install these packages in your Python environment before running the project.

## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies as mentioned in the "Dependencies" section.
3. Run `premier_league_data_scrape.py` to scrape and clean the data.
4. Open and run the Jupyter Notebook `premier_league_prediction.ipynb` to build and evaluate the prediction models.

Please note that this project serves as an educational resource and for entertainment purposes. Predicting football match outcomes accurately is a challenging task, and the models' performance may vary based on various factors, including the quality of the data and the features used.

Feel free to explore and extend this project to improve the models and enhance your understanding of machine learning in sports analytics.

