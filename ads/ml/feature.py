from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Feature(BaseEstimator, TransformerMixin):

    def __init__(self, name, cols=None, steps=None, dtype=np.float32):
        self.name = name
        self.dtype = dtype
        self.cols = cols
        if self.cols is None:
            self.cols = [name]
        self.steps = steps
        if self.steps is None:
            self.steps = []

    def fit(self, x, y=None):
        transformed = x[self.cols].copy()
        for step in self.steps:
            transformed = step.fit_transform(transformed, y)
        return self

    def transform(self, x):
        transformed = x[self.cols].copy()
        for step in self.steps:
            transformed = step.transform(transformed)
        assert np.isnan(transformed).sum() == 0, "Transformed contains nan"
        assert np.isinf(transformed).sum() == 0, "Transformed contains inf"
        return transformed.astype(self.dtype)

    def set_params(self, **kwargs):
        for attr, val in kwargs.items():
            self.__setattr__(attr, val)

    def __repr__(self, **kwargs):
        s = self.name + ":\n\t" + "\n\t".join([step.__repr__(N_CHAR_MAX=7000) for step in self.steps])
        return s


class FeatureDescriptor:
    def __init__(self, name, cols=None, wts=(1,)):
        self.name = name
        self.cols = cols
        self.wts = wts
        self._steps = None

    def __process_step(self, items):
        ret = []
        for item, param_grid in items:
            for kwargs in ParameterGrid(param_grid):
                instance = item()
                instance.set_params(**kwargs)
                ret.append(instance)
        return ret

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, vals):
        self._steps = vals

    @property
    def steps_space(self):
        processed_steps = [self.__process_step(step) for step in self.steps]
        return list(product(*processed_steps))


class NumFeatureDescriptor(FeatureDescriptor):
    def __init__(self, name, cols=None, wts=(1,)):
        super().__init__(name, cols, wts)
        self.steps = [
            [(SimpleImputer, {'strategy': ['median']})],
            [(StandardScaler, {})],
        ]


class CatFeatureDescriptor(FeatureDescriptor):
    def __init__(self, name, cols=None, categories=None, wts=(1,), fill_value='unk_val'):
        super().__init__(name, cols, wts)
        self.steps = [
            [(SimpleImputer, {'strategy': ['constant'], 'fill_value': [fill_value]})],
            [(OneHotEncoder, {'sparse': [False], 'categories': [[categories + [fill_value]]]})],
            [(StandardScaler, {})],
        ]
