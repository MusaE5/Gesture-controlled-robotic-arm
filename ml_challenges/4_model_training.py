import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("/home/moosey/gesture-arm/gesture_data.csv", header=None)
df.columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'label']

WINDOW_SIZE = 40
features = []
labels = []

for i in range(0, len(df)- WINDOW_SIZE, 20):
    window = df.iloc[i: i + WINDOW_SIZE]
    feature_vector = []
    for axis in ('Ax','Ay','Az','Gx','Gy','Gz'):
        mean = window[axis].mean()
        std = window[axis].std()

        feature_vector.append(mean)
        feature_vector.append(std)

    features.append(feature_vector)
    labels.append(window['label'].mode()[0])

x = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(X_train,y_train)


y_pred = model.predict(X_test)

print("🔍 Classification Report:\n", classification_report(y_test, y_pred))
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

