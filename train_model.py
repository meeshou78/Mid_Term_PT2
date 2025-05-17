# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Feature columns
X = df[["math score", "reading score", "writing score"]]

# Target encoding
df["race/ethnicity"] = df["race/ethnicity"].astype("category")
y = df["race/ethnicity"].cat.codes

# Save the mapping for decoding later
mapping = dict(enumerate(df["race/ethnicity"].cat.categories))
joblib.dump(mapping, "race_mapping.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, clf.predict(X_test)))

# Save model
joblib.dump(clf, "model.pkl")
