# Disaster Response Pipeline Project

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Instructions](#instructions)
- [Project Structure](#project-structure)
- [Evaluation](#evaluation)

## Project Overview
The **Disaster Response Pipeline** project is designed to build a machine learning model that classifies messages related to disasters. This application aims to assist humanitarian organizations in responding to crises effectively by categorizing messages into relevant disaster response categories.

## Data
The project utilizes two primary datasets:
- disaster_messages.csv: Contains messages sent during various disasters.
- disaster_categories.csv: Contains categories corresponding to the disaster messages.
These datasets can be found in the data/ directory.

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Project Structure

```bash
disaster-response-pipeline/
├── app/
│   ├── templates/                    # HTML templates for the web app
│   └── run.py                        # Main application file that starts the Flask server
├── data/
│   ├── disaster_categories.csv        # Disaster categories dataset
│   ├── disaster_messages.csv          # Disaster messages dataset
│   ├── disaster_response.db           # Database for disaster response
│   ├── ETL Pipeline Preparation.ipynb  # Jupyter notebook for ETL pipeline preparation
│   └── process_data.py               # Python script for processing data
├── models/
│   ├── classifier.pkl                 # Trained machine learning model
│   ├── disaster_response.db           # Database for disaster response
│   ├── ML Pipeline Preparation.ipynb   # Jupyter notebook for ML pipeline preparation
│   └── train_classifier.py            # Python script for training the classifier
└── README.md                         # Project documentation
```

## Evaluation
- The model's performance is evaluated using metrics including accuracy, precision, recall, and F1-score to ensure effective classification of disaster messages.