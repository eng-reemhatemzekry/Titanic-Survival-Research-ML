import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from feature_engineering import feature_engineering
from preprocessing import build_preprocessor

os.makedirs("models", exist_ok=True)

# Load data
train_df = pd.read_csv("data/train.csv")

# Feature engineering
train_df = feature_engineering(train_df)

# Split
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

# Define features
numeric_features = ["Age", "Fare", "FamilySize", "FarePerPerson", "TicketGroup"]
categorical_features = ["Sex", "Embarked", "Title", "Deck", "Pclass"]

preprocessor = build_preprocessor(numeric_features, categorical_features)

# Model
model = XGBClassifier(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train
pipeline.fit(X, y)

# Save
joblib.dump(pipeline, "models/model.pkl")

print("Training completed and model saved!")
