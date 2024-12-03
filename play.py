import pandas as pd

ip = pd.read_csv("data/indian_pines.csv")
print(len(ip.columns))
print(len(ip["class"].unique()))
print(len(ip))