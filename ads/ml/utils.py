import glob
import itertools
import os
import sys
from importlib import util
from pathlib import Path

from ads.ml.feature import Feature


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
        # for steps in steps_space:
        #     x = data[cols]
        #     for step in steps:
        #         print(name, step)
        #         x = step.fit_transform(x)
        weights.append(wts)
        names.append(name)
    fes_steps.update({'fes__transformer_weights': [dict(zip(names, comb)) for comb in itertools.product(*weights)]})
    return features, fes_steps


def fit_transform_feature(name, data, fe_dir):
    fe_file = glob.glob(os.path.join(fe_dir, '*', f'{name}.py'))[0]
    feature_module = get_feature_as_module(fe_file)
    steps = get_steps_space(feature_module)['steps_space'][0]
    feature = Feature(name=name, steps=steps)
    x = feature.fit_transform(data)
    return x
