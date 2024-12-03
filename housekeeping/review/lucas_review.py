import pandas as pd
import os

root = "../../data"
for dataset in os.listdir(root):
    if not dataset.startswith("lucas"):
        continue
    df = pd.read_csv(os.path.join(root, dataset))
    print("=======================")
    print(dataset)
    print("=======================")
    print("Columns",len(df.columns))
    print("Rows",len(df))
    print("Min SOC",df["oc"].min())
    print("Max SOC",df["oc"].max())