import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Create folders if not exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------
# STEP 1: CREATE SYNTHETIC DATA
# -------------------------------

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'age': np.random.randint(22, 50, n),
    'experience': np.random.randint(1, 20, n),
    'department': np.random.choice(['IT', 'HR', 'Sales'], n),
    'salary': np.random.randint(30000, 120000, n),
    'training_hours': np.random.randint(5, 100, n),
    'projects': np.random.randint(1, 10, n),
    'feedback_score': np.random.uniform(1, 5, n)
})

# Target logic
conditions = [
    (df['projects'] > 6) & (df['feedback_score'] > 4),
    (df['projects'] > 3)
]

choices = ['High', 'Medium']
df['performance'] = np.select(conditions, choices, default='Low')

# Save dataset
df.to_csv("data/employee_data.csv", index=False)

# -------------------------------
# STEP 2: ENCODING
# -------------------------------

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(columns=['performance_Low'])
y = df_encoded['performance_Low']

# -------------------------------
# STEP 3: TRAIN MODEL
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

# -------------------------------
# STEP 4: EVALUATION
# -------------------------------

y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# STEP 5: SAVE MODEL + COLUMNS
# -------------------------------

joblib.dump(model, "models/model.pkl")
joblib.dump(X.columns.tolist(), "models/columns.pkl")

print("✅ Model + Columns saved successfully!")