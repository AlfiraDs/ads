from ads.ml.preprocessing import BasePreprocessor
from sklearn.impute import SimpleImputer


class ValueFiller(BasePreprocessor):
    def __init__(self, strategy, fill_value=0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = None
        self.fitted = False

    def fit(self, x, y):
        self.imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        self.imputer.fit(x.reshape(-1, 1))
        self.fitted = True

    def transform(self, x):
        assert self.fitted
        return self.imputer.transform(x.reshape(-1, 1))[:, 0]
