import pandas as pd

data = {
    'algorithm': ['svr', 'linear', 'svr', 'tree', 'svr'],
    'r2': [0.85, 0.78, 0.88, 0.75, 0.90],
    'rmse': [0.12, 0.15, 0.11, 0.14, 0.10],
    'train_r2': [0.87, 0.76, 0.89, 0.78, 0.92],
    'train_rmse': [0.11, 0.14, 0.10, 0.13, 0.09]
}

df = pd.DataFrame(data)

result = df[df['algorithm'] == 'svr'].groupby('algorithm')[['r2', 'rmse', 'train_r2', 'train_rmse']].mean().reset_index()
result = result.iloc[0].to_dict()
print(result)
