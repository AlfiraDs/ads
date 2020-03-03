import numpy as np

from ads.ml.preprocessing import BasePreprocessor


class ColumnPreprocessor(BasePreprocessor):
    def __init__(self, nan_filler=BasePreprocessor(), encoder=BasePreprocessor(), scaler=BasePreprocessor(),
                 transformer=BasePreprocessor()):
        self.nan_filler = nan_filler
        self.encoder = encoder
        self.normalizer = scaler
        self.transformer = transformer

    def fit(self, x, y):
        for attr_name in [attr_name for attr_name in self.__dict__ if not attr_name.startswith("__")]:
            x = getattr(self, attr_name).fit_transform(x, y)

    def transform(self, x):
        for attr_name in [attr_name for attr_name in self.__dict__ if not attr_name.startswith("__")]:
            x = getattr(self, attr_name).transform(x)
        return x


class Preprocessor(BasePreprocessor):

    def __init__(self):
        """
        Applies transformation for each col_name: encoder.
        """
        self.steps = dict()
        self.transformed = None

    def add(self, col_name, **kwargs):
        """
        Add column with necessary steps to apply
        :param col_name: str
        :param kwargs: arguments to ColumnPreprocessor
        """
        self.steps[col_name] = ColumnPreprocessor(**kwargs)

    def fit(self, data, y):
        """
        :param data: DataFrame
        :param y: Series
        """
        for col_name, preprocessor in self.steps.items():
            preprocessor.fit(data[col_name].values, y.values)

    def transform(self, data):
        """
        :param data: DataFrame
        """
        transformed = None
        for col_name, preprocessor in self.steps.items():
            arr = preprocessor.transform(data[col_name].values)
            if len(arr.shape) == 1:
                arr = arr.reshape(-1, 1)
            transformed = np.hstack([transformed, arr]) if transformed is not None else arr
        return transformed
