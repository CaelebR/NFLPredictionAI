# NFL Game Prediction Model
An ensemble machine learning model that predicts NFL game outcomes using historical team statistics and game data.
## Overview
This project uses an ensemble of four machine learning algorithms (Random Forest, XGBoost, Decision Tree, and Logistic Regression) to predict NFL game winners with approximately 65% accuracy.
## Features

Ensemble Learning: Combines predictions from 4 different models using soft voting
Real-time Data: Uses nflreadpy to fetch current NFL season data
Feature Engineering: Calculates 24+ features including team statistics, advantages, and game context
Easy Predictions: Simple command-line interface for predicting any matchup
Model Persistence: Save and load trained

# Setup with Conda
bash# Clone the repository
git clone https://github.com/yourusername/NFL_prediction.git
cd NFL_prediction

## Create conda environment
conda env create -f environment.yml

## Activate environment
conda activate nfl_prediction

# Usage
## 1. Train the Model
Train on multiple seasons for better accuracy:
bashcd src
python train.py --seasons 2019 2020 2021 2022 2023 2024
Training Options:
bash# Use more test data
python train.py --seasons 2020 2021 2022 2023 --test-size 0.3

Use stacking ensemble instead of voting
python train.py --seasons 2020 2021 2022 2023 --model-type stacking

Force fresh data download
python train.py --seasons 2020 2021 2022 2023 --no-cache

Custom output directory
python train.py --seasons 2020 2021 2022 2023 --output-dir trained_models
## 2. Make Predictions
Predict upcoming week:
bashpython predict.py --season 2025 --week 14
Predict specific matchup:
bashpython predict.py --season 2025 --home-team KC --away-team BUF
Save predictions to CSV:
bashpython predict.py --season 2025 --week 14 --output predictions/week14.csv
## Additional options:
## Playoff game
python predict.py --season 2025 --home-team KC --away-team BUF --playoff

## Neutral site
python predict.py --season 2025 --home-team KC --away-team BUF --neutral

## Use custom model
python predict.py --season 2025 --week
