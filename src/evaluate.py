import pandas as pd
import joblib
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score
)

from feature_engineering import feature_engineering

# Load model
model = joblib.load("models/model.pkl")

# Load data
df = pd.read_csv("data/train.csv")
df = feature_engineering(df)

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Predict
preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]

# Metrics
print(classification_report(y, preds))
print("ROC-AUC:", roc_auc_score(y, probs))

# Confusion matrix
cm = confusion_matrix(y, preds)
ConfusionMatrixDisplay(cm).plot()
plt.show()

# Probability distribution
sns.histplot(probs, bins=30, kde=True)
plt.title("Prediction Confidence")
plt.show()
