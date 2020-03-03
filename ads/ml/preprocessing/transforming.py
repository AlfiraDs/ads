from ads.ml.preprocessing import BasePreprocessor


class DifferenceBetweenBaseNumber(BasePreprocessor):
    def __init__(self, base_number=None):
        self.base_number = base_number

    def fit(self, x, y=None):
        if not self.base_number:
            self.base_number = x.max()
        return self

    def transform(self, x):
        return self.base_number - x
