import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.signal import savgol_filter
from kennard_stone import train_test_split as ks_split
import matplotlib.pyplot as plt

cols = []
spec = 400
while spec <= 499.5:
    val = spec
    if int(val) == val:
        val = int(val)
    val = str(val)
    cols.append(val)
    spec = spec+0.5

df = pd.read_csv('../../data/min_lucas.csv')
df = df.drop(columns=cols).reset_index(drop=True).copy()
reflectance_cols = [col for col in df.columns if col != 'oc']
X = df[reflectance_cols].values

shortlisted = [100,101,102,103,104]
original_signal = X[shortlisted]

y = df['oc'].values
X_smooth = savgol_filter(X, window_length=41, polyorder=2, deriv=1, axis=1)

p1_signal = X_smooth[shortlisted]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_smooth)
mean_pca = np.mean(X_pca, axis=0)
cov_pca = np.cov(X_pca, rowvar=False)

mahal_distances = np.array([mahalanobis(x, mean_pca, np.linalg.inv(cov_pca)) for x in X_pca])
df_filtered = df.copy()
df_filtered['is_outlier'] = (mahal_distances > np.percentile(mahal_distances, 95)).astype(int)

p2_signal = X[100]


scalers = {col: MinMaxScaler() for col in reflectance_cols}
X_scaled = df_filtered[reflectance_cols].apply(lambda col: scalers[col.name].fit_transform(col.values.reshape(-1, 1)).flatten(), axis=0).to_numpy()


scaler_y = RobustScaler()
y_scaled = scaler_y.fit_transform(df_filtered[['oc']])

df_filtered_scaled = pd.DataFrame(X_scaled, columns=reflectance_cols)
df_filtered_scaled['oc'] = y_scaled


X_train, X_test, y_train, y_test = ks_split(X_scaled, y_scaled.ravel(), test_size=0.25)

print(X_train.shape)
print(y_test.shape)

fig, axes = plt.subplots(2, 3, figsize=(12, 10))
colors = ['blue', 'green', 'red', 'purple', 'orange']
for i, idx in enumerate(shortlisted):
    axes[0][0].plot(X[idx], color=colors[i], label=f'Signal {idx}')
axes[0][0].set_title('Original Signals')
axes[0][0].legend()

for i, idx in enumerate(shortlisted):
    axes[0][1].plot(X_scaled[idx], color=colors[i], label=f'Scaled Signal {idx}')
axes[0][1].set_title('MinMax Scaled Signals')
axes[0][1].legend()

for i, idx in enumerate(shortlisted):
    axes[0][2].plot(X_smooth[idx], color=colors[i], label=f'Scaled Signal {idx}')
    break
axes[0][2].set_title('MinMax Scaled Signals')
axes[0][2].legend()

axes[1][0].hist(y, bins=150, color='blue', alpha=0.7, edgecolor='black')
axes[1][0].set_title('Distribution of Original y')

axes[1][1].hist(y_scaled, bins=150, color='green', alpha=0.7, edgecolor='black')
axes[1][1].set_title('Distribution of Scaled y')

plt.show()
