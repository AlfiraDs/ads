import glob
import inspect
import itertools
import os
import sys
from importlib import util
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler


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
            if len(inspect.getfullargspec(step.fit_transform).args) > 2:
                transformed = step.fit_transform(transformed, y)
            else:
                transformed = step.fit_transform(transformed)
        return self

    def transform(self, x):
        transformed = x[self.cols].copy()
        for step in self.steps:
            transformed = step.transform(transformed)
            if len(transformed.shape) != 2:
                transformed = transformed.reshape(-1, 1)
        assert pd.isna(transformed).sum() == 0, "Transformed contains nan"
        assert pd.isna(
            pd.DataFrame(transformed).replace([np.inf, -np.inf], np.nan)).sum().sum() == 0, "Transformed contains inf"
        return transformed.astype(self.dtype)

    def set_params(self, **kwargs):
        for attr, val in kwargs.items():
            self.__setattr__(attr, val)

    def __repr__(self, **kwargs):
        s = self.name + ":\n\t" + "\n\t".join([step.__repr__() for step in self.steps])
        return s


class FeatureDescriptor:
    def __init__(self, name, cols=None, wts=(1,)):
        self.name = name
        self.cols = cols
        self.wts = wts
        self._steps = []

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
            # [(OneHotEncoder, {'sparse': [False], 'categories': [[categories + [fill_value]]]})],
            [(OrdinalEncoder, {'categories': [[categories + [fill_value]]]})],
            [(StandardScaler, {})],
            # [(MinMaxScaler, {})],
        ]


def get_feature_as_module(f):
    spec = util.spec_from_file_location('*', f)
    feature_module = util.module_from_spec(spec)
    spec.loader.exec_module(feature_module)
    return feature_module


def get_steps_space(feature_module):
    return {
            'name': feature_module.fd.name,
            'cols': feature_module.fd.cols,
            'wts': feature_module.fd.wts,
            'steps_space': feature_module.fd.steps_space,
        }


def steps_spaces_fn(features_dir_path: str):
    steps_spaces = {}
    for f in list(Path(features_dir_path).rglob("*.py")):
        f = str(f)
        feature_name = os.path.basename(f).split('.')[0]
        if '__' in f: continue
        feature_module = get_feature_as_module(f)
        try:
            feature_module.fd
        except AttributeError as e:
            print(f'Skipping feature {feature_name}', file=sys.stderr)
            continue
        steps_spaces[feature_name] = get_steps_space(feature_module)
    return steps_spaces


def get_search_space_fit_features(feature_dir):
    steps_spaces = steps_spaces_fn(feature_dir)
    features = []
    fes_steps = dict()
    weights = []
    names = []
    for module_name, steps_space_dict in steps_spaces.items():
        name = steps_space_dict['name']
        steps_space = steps_space_dict['steps_space']
        cols = steps_space_dict['cols']
        wts = steps_space_dict['wts']
        features.append((name, Feature(name, cols)))
        fes_steps[f'fes__{name}__steps'] = steps_space
        weights.append(wts)
        names.append(name)
    fes_steps.update({'fes__transformer_weights': [dict(zip(names, comb)) for comb in itertools.product(*weights)]})
    return features, fes_steps


def fit_transform_feature(feature_name, data, feature_dir):
    fe_file = list(Path(feature_dir).rglob(f'{feature_name}.py'))[0]
    feature_module = get_feature_as_module(fe_file)
    steps = get_steps_space(feature_module)['steps_space'][0]
    feature = Feature(name=feature_name, steps=steps)
    x = feature.fit_transform(data)
    return x
