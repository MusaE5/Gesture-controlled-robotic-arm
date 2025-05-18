import pandas as pd

# Load gesture data
df = pd.read_csv("gesture_data.csv", header=None)
df.columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'label']

WINDOW_SIZE = 40
labels = []

for i in range(0, len(df) - WINDOW_SIZE, 20):  # use step=20 if you trained like this
    window = df.iloc[i:i + WINDOW_SIZE]
    label = window['label'].mode()[0]  # most common label in window
    labels.append(label)

# Print label counts
print(pd.Series(labels).value_counts())
