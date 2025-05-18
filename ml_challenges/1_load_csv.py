import pandas as pd

df = pd.read_csv("/home/moosey/gesture-arm/gesture_data.csv", header= None)

print("First 5\n")
print(df.head())

print("Data types\n")
print(df.dtypes)

print("Label uniqueness\n")
print(df[6].unique())
