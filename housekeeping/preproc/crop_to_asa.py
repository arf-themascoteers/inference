import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.signal import savgol_filter
from kennard_stone import train_test_split as ks_split

cols = []
spec = 400
while spec <= 499.5:
    val = spec
    if int(val) == val:
        val = int(val)
    val = str(val)
    cols.append(val)
    spec = spec+0.5

df = pd.read_csv('../../data/lucas_crop.csv')
df = df.drop(columns=cols).reset_index(drop=True).copy()
reflectance_cols = [col for col in df.columns if col != 'oc']

X = df[reflectance_cols].values
y = df['oc'].values
X_smooth = savgol_filter(X, window_length=41, polyorder=2, deriv=1, axis=1)
df.loc[:, reflectance_cols] = X_smooth

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_smooth)
mean_pca = np.mean(X_pca, axis=0)
cov_pca = np.cov(X_pca, rowvar=False)

mahal_distances = np.array([mahalanobis(x, mean_pca, np.linalg.inv(cov_pca)) for x in X_pca])
outliers = mahal_distances > np.percentile(mahal_distances, 95)
df_filtered = df[~outliers]

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(df_filtered[reflectance_cols])
scaler_y = RobustScaler()
y_scaled = scaler_y.fit_transform(df_filtered[['oc']])

df_filtered_scaled = pd.DataFrame(X_scaled, columns=reflectance_cols)
df_filtered_scaled['oc'] = y_scaled
df_filtered_scaled.to_csv('../../data/lucas_crop_asa.csv', index=False)

# X_train, X_test, y_train, y_test = ks_split(X_scaled, y_scaled.ravel(), test_size=0.25)
#
# print(X_train.shape)
# print(y_test.shape)
