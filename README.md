# Loan Approval Prediction System

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20scikit--learn-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

A full-stack machine learning application that predicts loan approval based on applicant and financial details.
Built with **Flask**, **HTML/CSS/JS**, and **XGBoost**, it delivers real-time results with integrated visualization and model explainability.

---

## Table of Contents

* [Features](#features)
* [Tech Stack](#tech-stack)
* [Installation](#installation)
* [About Virtual Environments](#about-virtual-environments)
* [Model Training](#model-training)
* [Feature Importance](#feature-importance)
* [Data Imbalance Analysis](#data-imbalance-analysis)
* [Running the App](#running-the-app)
* [Project Structure](#project-structure)
* [Example Workflow](#example-workflow)
* [Future Enhancements](#future-enhancements)
* [License](#license)

---

## Features

* Machine learning pipeline using **XGBoost** and **Random Forest** for accurate classification.
* Automated data cleaning, encoding, and balancing of training samples.
* Explainable AI integration with **SHAP** for model interpretation.
* Responsive and interactive web interface built with **HTML**, **CSS**, and **JavaScript**.
* **Flask REST API** for connecting prediction results to the frontend.

---

## Tech Stack

| Layer         | Tools Used                     |
| ------------- | ------------------------------ |
| Backend       | Flask, scikit-learn, XGBoost   |
| Frontend      | HTML5, CSS3, JavaScript        |
| ML Toolkit    | pandas, NumPy, SHAP            |
| Visualization | Matplotlib, SHAP summary plots |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction
```

### About Virtual Environments

A **virtual environment** is an isolated Python workspace that keeps project dependencies separate from your system Python installation.
This helps avoid version conflicts between projects.

For example:

* Project A may need **Flask 2.0**
* Project B may need **Flask 3.0**

Creating a virtual environment ensures each project uses the correct version without interference.

Create and activate one:

```bash
python -m venv venv
source venv/bin/activate     # On Windows use venv\Scripts\activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

Ensure a dataset file named `loans.csv` is available in the project root directory.

---

## Model Training

Run the training script:

```bash
python train_model.py
```

This script:

* Loads and preprocesses the dataset
* Performs hyperparameter tuning with **GridSearchCV**
* Prints metrics (accuracy, F1-score, AUC)
* Generates **SHAP plots** for feature impact
* Saves the trained model as `model.pkl`

---

## Feature Importance

Display key variables and their impact:

```bash
python feature_importance.py
```

This shows top predictors like `CreditHistory`, `ApplicantIncome`, and `LoanAmount`.

---

## Data Imbalance Analysis

Check class balance between approved and rejected applications:

```bash
python check_imbalance.py
```

Displays class count ratios and percentages to identify potential bias.

---

## Running the App

Start the backend server:

```bash
python manage.py runserver
```

Then open the file `index.html` or go to:

```
http://localhost:8000
```

Enter applicant details to generate predictions.

---

## Project Structure

```
loan-approval-prediction/
├── index.html                # Main application UI
├── styles.css                # Layout and design
├── script.js                 # Browser-side JS logic
├── train_model.py            # Model training logic
├── feature_importance.py     # Feature ranking
├── check_imbalance.py        # Balance diagnostics
├── requirements.txt          # Dependencies
├── model.pkl                 # Trained model
```

---

## Example Workflow

1. Place dataset (`loans.csv`) in the directory.
2. Run model training with `train_model.py`.
3. Start the Flask backend using `manage.py`.
4. Access the UI in your browser.
5. Analyze output and visualize feature influence.

---

## Future Enhancements

* Add database support (PostgreSQL or MongoDB)
* Extend model selection (Logistic Regression, LightGBM, etc.)
* Deploy with **AWS**, **Render**, or **Docker**
* Build a model comparison dashboard

---

## License

This project is distributed under the **MIT License**.
You may freely use and modify it for educational or research purposes.

---
