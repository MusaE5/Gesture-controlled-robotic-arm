import pandas as pd

df= pd.read_csv("/home/moosey/gesture-arm/gesture_data.csv", header = None)
df.columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'label']

WINDOW_SIZE = 40

features = []
labels = []

for i in range(0, len(df)- WINDOW_SIZE, WINDOW_SIZE):
    window = df.iloc[i:i+WINDOW_SIZE]
    feature_vector = []

    for axis in['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz' ]:
        mean = window[axis].mean()
        std = window[axis].std()

        feature_vector.append(mean)
        feature_vector.append(std)

    features.append(feature_vector)    

    label = window['label'].mode()[0]
    labels.append(label)

print(" Feature vector (first window):")
print(features[0])

print("\n Label for first window:", labels[0])

print(f"\n Total windows processed: {len(features)}")

    

