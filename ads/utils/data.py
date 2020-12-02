import os
from glob import glob
from typing import Iterable

import pandas as pd
from ads.utils.eda import nan_stat


class Data:
    def __init__(self,
                 data_dir: str = None,
                 target_col: str = 'tgt',
                 train_file_kwd: str = 'train',
                 test_file_kwd: str = 'test',
                 dtype: dict = None,
                 drop_rows: Iterable = None
                 ):
        data_dir = data_dir if data_dir is not None else os.path.join(os.getcwd(), 'data')
        train_file = glob(os.path.join(data_dir, f'*{train_file_kwd}*'))[0]
        test_file = glob(os.path.join(data_dir, f'*{test_file_kwd}*'))[0]
        self._drop_rows = drop_rows
        self._nans = [None, "None", '?']
        self._dtype = dtype
        self._train = self.__read_csv(train_file)
        self._train = self._train.drop(self._drop_rows, axis=0, errors='warn')
        self._test = self.__read_csv(test_file)
        self._data = pd.concat([self.train, self.test])
        self._y = self.train[target_col]
        self._train = self._train.drop(target_col, axis=1)
        print("train:", self.train.shape)
        print("test:", self.test.shape)
        print("y:", self.y.shape)
        print(nan_stat(self.data))

    def __read_csv(self, file_path):
        df = pd.read_csv(file_path, na_values=self._nans, keep_default_na=True, encoding="utf-8", dtype=self._dtype)
        return df

    @property
    def data(self):
        return self._data

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def y(self):
        return self._y
