import pandas as pd
import numpy as np

from ads.ml.preprocessing import BasePreprocessor
from ads.ml.preprocessing import StatTargetOrderEncoder
from ads.ml.preprocessing import Preprocessor
from ads.ml.preprocessing import RangeScaler

if __name__ == "__main__":
    data = pd.DataFrame({
        "x1": np.random.randint(1, 10, size=100) * 10,
        "x2": np.random.randint(1, 10, size=100) * 10
    })
    y = pd.Series(np.random.randn(100) * 1000, name="y")
    pp = Preprocessor()
    pp.add("x1", nan_filler=BasePreprocessor(),
           encoder=StatTargetOrderEncoder(),
           scaler=BasePreprocessor(),
           transformer=BasePreprocessor())
    pp.add("x2", nan_filler=BasePreprocessor(),
           encoder=StatTargetOrderEncoder(),
           scaler=RangeScaler(feature_range=(0, 1)),
           transformer=BasePreprocessor())
    transformed = pp.fit_transform(data, y)
    print()
