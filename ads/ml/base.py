import numpy as np
import pandas as pd
import hashlib
from sklearn.base import BaseEstimator, TransformerMixin


class EmptyStep(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.zeros_like(x)


class BasePreprocessor(BaseEstimator, TransformerMixin):

    def set_params(self, **kwargs):
        for attr, val in kwargs.items():
            self.__setattr__(attr, val)

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x

    def predict(self, x):
        return x


class Feature(BasePreprocessor):

    def __init__(self, name=None, columns=None, steps=None, dimension=None, dummy=False,
                 imputer=BasePreprocessor(),
                 processor=BasePreprocessor(),
                 transformer=BasePreprocessor(),
                 scaler=BasePreprocessor()):
        self.fitted = False
        self.dummy = dummy
        self.name = name
        self.dimension = dimension
        if columns is None:
            self.columns = [name]
        else:
            self.columns = columns
        if steps is None:
            self.steps = [imputer, processor, transformer, scaler]
        else:
            self.steps = steps

    def fit(self, x, y=None):
        if self.dummy or self.fitted:
            pass
        else:
            fitted = x
            for step in self.steps:
                # print(self.name, step)
                fitted = step.fit_transform(fitted, y)
            self.fitted = True
        return self

    def transform(self, x):
        if self.dummy: return np.zeros_like(x)
        transformed = x
        for step in self.steps:
            # print(self.name, step)
            transformed = step.transform(transformed)
        assert np.isnan(transformed).sum() == 0, "Transformed contains nan"
        assert np.isinf(transformed).sum() == 0, "Transformed contains inf"
        return transformed

    def __repr__(self, **kwargs):
        s = self.name + ":\n\t" + "\n\t".join([step.__repr__() for step in self.steps])
        return s


class Features(BasePreprocessor):

    @staticmethod
    def required_list_to_dict(features):
        return {feature.name: feature for feature in features}

    @staticmethod
    def optional_list_to_dict(features):
        optional = {}
        for options in features:
            options_name = set([feature.name for feature in options.categories])
            assert len(options_name) == 1, "There are different names for feature options: %s" % options_name
            optional["fes__" + options_name.pop()] = options
        return optional

    def __init__(self, features):
        """
        :param features: dictionary: {feature_name: base.Feature()}
        """
        self.features = features
        self.data_fn = None

    def set_params(self, **kwargs):
        for attr, val in kwargs.items():
            if attr == "dummy": continue
            self.features[attr] = val

    def fit(self, x_idxs, y=None):
        for feature_name, feature in self.features.items():
            if feature.fitted:
                pass
            else:
                x = self.data_handler(x_idxs, feature)
                feature.fit(x)
        return self

    def transform(self, x_idxs):
        transformed = None
        for feature_name, feature in self.features.items():
            x = self.data_handler(x_idxs, feature)
            arr = feature.transform(x)
            transformed = np.hstack([transformed, arr]) if transformed is not None else arr
        return transformed

    def data_handler(self, x, feature=None):
        if self.data_fn:
            x = self.data_fn(feature.columns)[x.ravel()]
        else:
            x = x[feature.columns]
        return x

    def __repr__(self):
        repr = "\n".join([feature.__repr__() for name, feature in self.features.items()])
        hash_object = hashlib.md5(repr.encode())
        # print(hash_object.hexdigest())
        return repr
