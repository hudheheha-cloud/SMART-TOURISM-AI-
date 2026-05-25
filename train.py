import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_excel(r"E:\Pakistan_Unified_Tourism_Master_Dataset.xlsx",
                   sheet_name="Destination Metrics",
                   skiprows=4)

df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df = df[df['Destination'].str.contains("Average Metric", na=False) == False]

# -----------------------------
# FEATURES / TARGET
# -----------------------------
X = df[['Budget Index','Weather Profile','Trip Style','Peak Month']]
y = df['Destination']

categorical_features = X.columns.tolist()

# -----------------------------
# PREPROCESSOR (NO MANUAL COLUMNS EVER AGAIN)
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# -----------------------------
# MODEL PIPELINE
# -----------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# -----------------------------
# TRAIN
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# -----------------------------
# SAVE ONLY ONE MODEL FILE
# -----------------------------
joblib.dump(model, "tourism_model.pkl")

print("Model saved successfully")