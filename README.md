# Phishing Email Detection Model

## Description

This project uses Machine Learning and Scikit-learn to classify messages as Phishing or Safe.

The model uses TF-IDF text features along with additional features such as:

- Number of URLs
- Number of URLs containing IP addresses
- Suspicious keyword count
- Message length
- Uppercase character ratio

## Machine Learning Model

The project uses Logistic Regression for classification.

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- SciPy

## Features

- Dataset loading
- Text preprocessing
- TF-IDF feature extraction
- URL feature extraction
- Phishing/Safe classification
- Accuracy calculation
- Classification report
- Confusion matrix
- Prediction of new emails

## Output

The model displays:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Prediction and confidence for new emails

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
