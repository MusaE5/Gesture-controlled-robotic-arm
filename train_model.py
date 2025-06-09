import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv("gesture_data.csv", header=None)
df.columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'label']

# Group data into windows (20  samples per gesture)
WINDOW_SIZE = 20
features = []
labels = []

for i in range(0, len(df) - WINDOW_SIZE, 20):
    window = df.iloc[i:i+WINDOW_SIZE]
    label = window['label'].mode()[0]  # Most frequent label in the window

    # Extract features: mean and std of each sensor axis
    feature_vector = []
    for axis in ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']:
        feature_vector.append(window[axis].mean())
        feature_vector.append(window[axis].std())

    features.append(feature_vector)
    labels.append(label)

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print(" Model saved as model.pkl")
