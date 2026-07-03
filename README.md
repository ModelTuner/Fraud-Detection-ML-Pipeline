# Fraud Detection using Machine Learning

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning. The main challenge with this dataset is that it is highly imbalanced, so I used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data before building the models.

The project includes data preprocessing, visualization, model training, hyperparameter tuning, and performance evaluation. Instead of relying on accuracy, the models are evaluated using **Precision**, **Recall**, and **ROC-AUC**, which are more appropriate for fraud detection problems.

---

## Features

* Data Exploration (EDA)
* Data Visualization
* Feature Scaling
* Handling Class Imbalance using SMOTE
* Logistic Regression Model
* Random Forest Model
* Hyperparameter Tuning with GridSearchCV
* Model Evaluation
* Feature Importance Analysis

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)

---

## Project Structure

```text
Fraud-Detection-ML-Pipeline/
│
├── fraud_detection.py
├── README.md
├── requirements.txt
├── creditcard.csv (Not Included)
├── model_comparison.csv
├── feature_importance.csv
└── Output Images
```

---

## How to Run

1. Clone this repository.

2. Install the required libraries.

```bash
pip install -r requirements.txt
```

3. Place the `creditcard.csv` dataset in the project folder.

4. Run the project.

```bash
python fraud_detection.py
```

---

## Evaluation Metrics

The models are evaluated using:

* Precision
* Recall
* ROC-AUC Score

Since the dataset is highly imbalanced, accuracy is not considered the primary evaluation metric.

---

## Results

After comparing different models, the tuned Random Forest model gave the best overall performance. The project also generates confusion matrices, ROC curves, and feature importance graphs to better understand the model's performance.

---

## Future Improvements

* Try XGBoost and LightGBM
* Build a web application using Flask or Streamlit
* Deploy the model for real-time predictions

---

## Author

**Kush Sharma**

Machine Learning & Data Science Enthusiast
