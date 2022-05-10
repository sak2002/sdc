import pandas as pd
from numba import njit


@njit
def series_transform():
    s = pd.Series([20, 21, 12],
                  index=['London', 'New York', 'Helsinki'])

    def sum(x):
        return x + 1

    return s.transform(sum)


@njit
def sum(x):
    return x + 1


@njit
def square(x):
    return x ** 2


@njit
def series_transform_tuple():
    s = pd.Series([20, 21, 12],
                  index=['London', 'New York', 'Helsinki'])
    func = (sum, square)
    return s.transform(func, 1, 2)


print(series_transform())
print(series_transform_tuple())
