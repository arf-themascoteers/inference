# import pandas as pd
#
# ip = pd.read_csv("data/indian_pines.csv")
# print(len(ip.columns))
# print(len(ip["class"].unique()))
# print(len(ip))


import numpy as np

X = 200
n = 300

linearly_spaced_integers = np.linspace(0, X, n, dtype=int).tolist()
print(linearly_spaced_integers)