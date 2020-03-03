from ads.ml.preprocessing import BasePreprocessor
from sklearn.preprocessing import MinMaxScaler


class Scaler(BasePreprocessor):
    pass


class RangeScaler(BasePreprocessor):
    def __init__(self, feature_range=(0, 1)):
        self.scaler = None
        self.fitted = False
        self.feature_range = feature_range

    def fit(self, x, y):
        self.scaler = MinMaxScaler(feature_range=self.feature_range).fit(x.reshape(-1, 1), y)
        self.fitted = True

    def transform(self, x):
        assert self.scaler and self.fitted, "need to fit first"
        return self.scaler.transform(x.reshape(-1, 1))[:, 0]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.scaler.transform(x.reshape(-1, 1))[:, 0]
