import pandas as pd

df= pd.read_csv("/home/moosey/gesture-arm/gesture_data.csv", header = None)
df.columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'label']

WINDOW_SIZE = 40
num_windows = 0

for i in range(0, len(df)-WINDOW_SIZE, WINDOW_SIZE):
    window = df.iloc[i:i + WINDOW_SIZE]
    num_windows += 1

    if i == 400:
        print(" Second window:")
        print(window.head())

# Report number of windows created
print(f"\n Total windows created: {num_windows}")
