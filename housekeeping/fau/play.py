import numpy as np

data = [10, 20, 30, 40]

pop_std = np.std(data, ddof=0)
sample_std = np.std(data, ddof=1)

print(pop_std)
print(sample_std)