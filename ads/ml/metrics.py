import numpy as np
from math import sqrt

from sklearn.metrics import mean_squared_error


def rmse(model, X, y):
    predictions = model.predict(X)
    predictions[predictions <= 0] = y.min()
    se = sqrt(mean_squared_error(np.log(y), np.log(predictions)))
    # se = sqrt(mean_squared_error(y, predictions))
    return se
