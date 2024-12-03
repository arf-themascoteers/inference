from algorithm import Algorithm
import torch
import torch.nn as nn
import numpy as np

class Algorithm_simple_classification(Algorithm):
    def __init__(self, dataset, train_x, train_y, test_x, test_y, target_size, class_size, fold, reporter, verbose):
        super().__init__(dataset, train_x, train_y, test_x, test_y, target_size, class_size, fold, reporter, verbose)

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True

        self.train_x = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        self.train_y = torch.tensor(train_y, dtype=torch.int32).to(self.device)
        self.test_x = torch.tensor(test_x, dtype=torch.float32).to(self.device)
        self.test_y = torch.tensor(test_y, dtype=torch.int32).to(self.device)
        self.indices = list(range(self.train_x.shape[1]))
        if self.target_size != self.train_x.shape[1] and self.target_size != -1:
            self.indices = np.linspace(0, self.train_x.shape[1]-1, self.target_size, dtype=int).tolist()
            self.train_x = self.train_x[:, self.indices]
            self.test_x = self.test_x[:, self.indices]

        self.criterion = torch.nn.CrossEntropyLoss()
        self.class_size = 1
        self.lr = 0.001
        self.total_epoch = 1000

        self.ann = nn.Sequential(
            nn.Linear(target_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Linear(32, class_size),
        )
        self.ann.to(self.device)
        self.reporter.create_epoch_report(self.get_name(), self.dataset.name, self.target_size, self.fold)

    def _fit(self):
        self.ann.train()
        self.write_columns()
        optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)
        y = self.train_y.type(torch.LongTensor).to(self.device)
        for epoch in range(self.total_epoch):
            optimizer.zero_grad()
            y_hat = self.predict_train()
            loss = self.criterion(y_hat, y)
            loss.backward()
            #print(loss.item())
            optimizer.step()
            self.report(epoch)
        return self

    def predict_train(self):
        return self.ann(self.train_x)

    def predict_test(self):
        return self.ann(self.test_x)

    def write_columns(self):
        if not self.verbose:
            return
        columns = ["epoch","r2","rmse","rpd","rpiq","train_r2","train_rmse","train_rpd","train_rpiq"] + [f"band_{index+1}" for index in range(self.target_size)]
        print("".join([str(i).ljust(20) for i in columns]))

    def report(self, epoch):
        if not self.verbose:
            return
        if epoch%10 != 0:
            return

        bands = self.indices

        train_y_hat = self.predict_train()
        test_y_hat = self.predict_test()

        r2, rmse, rpd, rpiq = self.calculate_metrics(self.test_y, test_y_hat)
        train_r2, train_rmse, train_rpd, train_rpiq = self.calculate_metrics(self.train_y, train_y_hat)

        self.reporter.report_epoch_bsdr(epoch, r2, rpd, rpiq, rmse, train_r2, train_rmse, train_rpd, train_rpiq, bands)
        cells = [epoch, r2, rmse, rpd, rpiq, train_r2, train_rmse, train_rpd, train_rpiq] + bands
        cells = [round(item, 5) if isinstance(item, float) else item for item in cells]
        print("".join([str(i).ljust(20) for i in cells]))


    def get_indices(self):
        return self.indices