# Phishing Email Detection Model

## Overview

A Machine Learning project that detects phishing emails using Natural Language Processing (NLP) and Scikit-Learn.

## Features

* Phishing Email Detection
* Safe Email Classification
* TF-IDF Vectorization
* Naive Bayes Classifier
* Accuracy Evaluation

## Dataset

* 82,486 email samples
* Labels:

  * 1 = Phishing
  * 0 = Safe

## Results

* Accuracy: 97.77%

Confusion Matrix:

[[7854, 81],
[287, 8276]]

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Joblib

## Run

Train Model:

python src/train.py

Predict:

python src/predict.py
