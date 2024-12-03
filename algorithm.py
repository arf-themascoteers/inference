from abc import ABC, abstractmethod
from sklearn.metrics import r2_score, mean_squared_error
import torch
import importlib
import numpy as np


class Algorithm(ABC):
    def __init__(self, dataset, train_x, train_y, test_x, test_y, target_size, fold, reporter, verbose):
        self.dataset = dataset
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        
        self.target_size = target_size
        self.fold = fold
        self.reporter = reporter
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_fold(self):
        self._fit()
        train_y_hat = self.predict_train()
        test_y_hat = self.predict_test()
        train_r2, train_rmse, train_rpd, train_rpiq = self.calculate_metrics(self.train_y, train_y_hat)
        r2, rmse, rpd, rpiq = self.calculate_metrics(self.test_y, test_y_hat)
        self.reporter.write_details(self.get_name(),self.dataset.name,self.target_size,
                                    r2, rmse, rpd, rpiq,
                                    train_r2, train_rmse,train_rpd, train_rpiq,
                                    self.fold, self.get_indices())

    def calculate_metrics(self, y_test, y_pred):
        y_test = Algorithm.convert_to_numpy(y_test.detach().cpu().numpy())
        y_pred = Algorithm.convert_to_numpy(y_pred.detach().cpu().numpy())

        r2 = r2_score(y_test, y_pred)
        #r2 = max(r2,0)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        #rmse = max(rmse,0)

        std_dev = np.std(y_test, ddof=1)
        rpd = std_dev/rmse
        #rpd = max(rpd, 0)

        iqr = np.percentile(y_test, 75) - np.percentile(y_test, 25)
        rpiq = iqr/rmse
        #rpiq = max(rpiq, 0)

        return round(r2, 2), round(rmse, 2), round(rpd, 2), round(rpiq, 2)

    @staticmethod
    def convert_to_numpy(t):
        if torch.is_tensor(t):
            return t.detach().cpu().numpy()
        return t

    @abstractmethod
    def _fit(self):
        pass

    def get_name(self):
        class_name = self.__class__.__name__
        name_part = class_name[len("Algorithm_"):]
        return name_part

    @staticmethod
    def create(name, dataset, train_x, train_y, test_x, test_y, target_size, fold, reporter, verbose):
        class_name = f"Algorithm_{name}"
        module = importlib.import_module(f"algorithms.algorithm_{name}")
        clazz = getattr(module, class_name)
        return clazz(dataset, train_x, train_y, test_x, test_y, target_size, fold, reporter, verbose)

    @abstractmethod
    def get_indices(self):
        pass

    @abstractmethod
    def predict_train(self):
        pass

    @abstractmethod
    def predict_test(self):
        pass