import pandas as pd

df = pd.read_csv("data/students.csv")

print("DATA:")
print(df)

print("\nSHAPE:")
print(df.shape)

print("\nCOLUMNS:")
print(df.columns)

print("\nINFO:")
print(df.info())

print("\nSTATISTICS:")
print(df.describe())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATES:")
print(df.duplicated().sum())

print("\nDATA TYPES:")
print(df.dtypes)