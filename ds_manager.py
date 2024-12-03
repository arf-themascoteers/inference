import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DSManager:
    def __init__(self, name="lucas",folds=1):
        self.name = name
        dataset_path = f"data/{self.name}.csv"
        df = pd.read_csv(dataset_path)
        self.folds = folds
        df = df.sample(frac=1).reset_index(drop=True)
        self.data = df.to_numpy()
        self.data = self.data[self.data[:, -1] != 0]
        self.data[:, -1] = self.data[:, -1]-1
        self.data[:,0:-1] = MinMaxScaler().fit_transform(self.data[:,0:-1])
        self.class_size = 1
        if self.is_classification():
            self.class_size = np.unique(self.data[:, -1]).size

    def get_k_folds(self):
        for i in range(self.folds):
            train_data, test_data = train_test_split(self.data, test_size=0.25, random_state=42+i)
            yield train_data[:,0:-1], train_data[:,-1], test_data[:,0:-1], test_data[:,-1]

    def is_classification(self):
        return self.name!="lucas"


if __name__ == '__main__':
    import numpy as np
    ds = DSManager(folds=1)
    for train_x, train_y, test_x, test_y in ds.get_k_folds():
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        print(np.min(train_y))
        print(np.max(train_y))
        print(np.max(train_y)-np.min(train_y))
