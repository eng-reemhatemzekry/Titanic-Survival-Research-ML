# Titanic Advanced ML Pipeline

## Overview

This project is an end-to-end machine learning pipeline for predicting passenger survival on the Titanic dataset from Kaggle.

The main goal of this project is to explore advanced machine learning techniques while following professional ML engineering practices such as:

- Feature Engineering
- Data Preprocessing
- Ensemble Learning
- Hyperparameter Tuning
- Explainable AI
- Cross Validation
- Error Analysis

The project is organized in a modular and clean structure similar to real-world ML projects.

---

# Project Structure

```bash
Titanic-Advanced-ML-Pipeline/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Explainability.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── models/
├── submissions/
├── reports/
│   ├── figures/
│   └── results/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Dataset

Dataset source:
https://www.kaggle.com/competitions/titanic

The dataset contains information about Titanic passengers such as:

- Age
- Gender
- Ticket class
- Fare
- Cabin
- Family members
- Survival status

Target variable:

```python
Survived
```

- 0 → Did not survive
- 1 → Survived

---

# Machine Learning Workflow

## 1. Exploratory Data Analysis (EDA)

Performed data analysis and visualization to understand:

- Missing values
- Feature distributions
- Survival patterns
- Correlations between features

Visualizations include:

- Count plots
- Heatmaps
- Histograms
- Survival analysis plots

---

## 2. Feature Engineering

Created new features to improve model performance, including:

- Title extraction from names
- Family size
- IsAlone feature
- Deck extraction
- Fare per person
- Ticket group size

---

## 3. Data Preprocessing

Implemented preprocessing pipelines using:

- Missing value imputation
- Feature encoding
- Feature scaling

Tools used:

- Pipeline
- ColumnTransformer
- OneHotEncoder
- StandardScaler

---

## 4. Models Used

The following machine learning models were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Stacking Ensemble

---

## 5. Hyperparameter Tuning

Used optimization techniques to improve model performance:

- RandomizedSearchCV
- Optuna

---

## 6. Cross Validation

Used Stratified K-Fold Cross Validation for more reliable evaluation and reduced overfitting.

---

## 7. Explainable AI (XAI)

Used SHAP to explain model predictions and feature importance.

Explainability techniques include:

- SHAP summary plots
- Feature importance analysis
- Individual prediction explanations

---

## 8. Error Analysis

Analyzed model mistakes to better understand:

- False positives
- False negatives
- Difficult passenger cases

---

# Technologies Used

## Programming Language

- Python

## Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- catboost
- optuna
- shap
- joblib

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Titanic-Advanced-ML-Pipeline.git
```

Move into the project directory:

```bash
cd Titanic-Advanced-ML-Pipeline
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run notebooks in the following order:

```bash
01_EDA.ipynb
02_Feature_Engineering.ipynb
03_Modeling.ipynb
04_Explainability.ipynb
```

---

# Results

| Model | Accuracy |
|---|---|
| Logistic Regression | TBD |
| Random Forest | TBD |
| XGBoost | TBD |
| LightGBM | TBD |
| CatBoost | TBD |
| Stacking Ensemble | TBD |

Final Kaggle score:

```python
TBD
```

---

# Future Improvements

Possible future improvements include:

- Advanced feature selection
- AutoML techniques
- Deep learning models
- Better ensemble strategies
- Model deployment

---

# Author

Reem Hatem

AI Engineer specializing in:
- Machine Learning
- Deep Learning
- Computer Vision

---

# License

This project is licensed under the MIT License.
