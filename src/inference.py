import pandas as pd
import joblib

from feature_engineering import feature_engineering

# Load model
model = joblib.load("models/model.pkl")

# Load test data
test_df = pd.read_csv("data/test.csv")

test_df = feature_engineering(test_df)

PassengerId = test_df["PassengerId"]

X_test = test_df.drop(columns=["PassengerId"])

# Predict
preds = model.predict(X_test)

# Submission
submission = pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": preds.astype(int)
})

submission.to_csv("submissions/submission.csv", index=False)

print("Submission file created!")
