from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from ads.ml.preprocessing import BasePreprocessor


class StatTargetOrderEncoder(BasePreprocessor):

    def __init__(self, statistic_fn=np.median, all_labels=None):
        self.statistic_fn = statistic_fn
        self.mapping = defaultdict(lambda: len(self.mapping))
        self.fitted = False

    def fit(self, x, y):
        # TODO: define final shapes of x and y
        assert not self.mapping, "An instance is already fitted."
        x = x.ravel()
        ordered_labels = pd.DataFrame(np.stack([x, y], axis=1))
        ordered_labels[1] = ordered_labels[1].astype(y.dtype)
        ordered_labels = ordered_labels.groupby(0)[1].agg(self.statistic_fn).sort_values()
        for label in ordered_labels.index:
            self.mapping[label] += 0
        self.fitted = True
        return self

    def transform(self, x):
        assert len(self.mapping) > 0, "self.mapping is empty. Call fit method first."
        assert self.fitted, "self.fitted is False. Call fit method first."
        transformed = pd.Series(x.reshape(-1)).map(self.mapping).values
        assert np.isfinite(transformed).all()
        return transformed


class SimpleEncoder(BasePreprocessor):

    def __init__(self):
        self.mapping = None

    def fit(self, x, y=None):
        if not self.mapping:
            self.mapping = {val: key for key, val in enumerate(np.unique(x))}
        return self

    def transform(self, x):
        """
        :param x: is a column vector
        :return: a column vector
        """
        assert self.mapping, "Call fit method first."
        transformed = np.vectorize(self.mapping.get)(x)
        return transformed


class AlphaOrderEncoder(BasePreprocessor):

    def __init__(self):
        self.encoder = LabelEncoder()
        self.fitted = False

    def fit(self, x, y):
        if not self.fitted:
            self.encoder.fit(x.ravel())
            self.fitted = True
        return self

    def transform(self, x):
        assert self.fitted, "Call fit method first."
        transformed = self.encoder.transform(x.ravel())
        return transformed


class LabelEncoderByMap(BasePreprocessor):

    def __init__(self, encoding_map):
        self.encoding_map = encoding_map
        self.fitted = False

    def fit(self, x, y):
        if not self.fitted:
            self.fitted = True
        return self

    def transform(self, x):
        assert self.fitted, "Call fit method first."
        transformed = pd.Series(x, dtype=str).map(self.encoding_map).values
        return transformed
