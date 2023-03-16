# Table of Contents
1. Installation
2. Project Motivation
3. Fle Descriptions
4. Results
5. Commands
6. Licensing, Authors, and Acknowledgements

# Installation
Libraries required for this project can be found in `requirements.txt`

# Project Motivation
This is the second project of Udacity Data Scientist Nano Degree program, aiming to practice our skills on using Scikit-learn's `Pipeline`. 
In this project, I trained a model that analyzes the content of the messages sent by people in disasters. 
The model returns the most possible categories of the messages, which will then be directed to appropriate emergent agencies.

# File Descriptions
```
|-- app
|     |-- templates
|     |     |-- master.html: this is the main page of the application
|     |     |-- go.html: this displays the classification result
|     |-- run.py: this is the python file that runs the application 
|     |-- mainpage.png: this is the snapshot of main page of the application
|     |-- resultpage.png: this is the snapshot of result page of the application
|-- data
|     |-- categories.csv: this contains all the categories of the messages
|     |-- messages.csv: this contains all the messages content
|     |-- DisasterResponse.db: this is the sqlite db file that contains cleaned data
|     |-- process_data.py: this is the python file that does the etl
|     |-- ETL Pipeline Preparation.ipynb
|-- models
|     |-- train_classifier.py: this is the python file that trains the classifier
|     |-- ML Ppeline Preparation.ipynb
```

# Results
- Below is a snapshot of the main page
![mainpage](https://github.com/anqi-guo/udacity-dsnd-project2/blob/main/app/mainpage.png)
- Below is a snapshot of the classification result page
![resultpage](https://github.com/anqi-guo/udacity-dsnd-project2/blob/main/app/resultpage.png)

# Commands
- process the data
```
python data/process_data.py data/messages.csv data/categories.csv sqlite:///data/DisasterResponse.db
```
- train the classifier
```
python models/train_classifier.py sqlite:///data/DisasterResponse.db models/model.pkl
```
- run the application
```
python app/run.py
```

# Licensing, Authors, and Acknowledgements
The data was provided by [Appen](https://appen.com/)