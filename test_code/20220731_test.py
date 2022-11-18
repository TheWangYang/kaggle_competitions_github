import pandas as pd
from pandas import Series

x = Series([True, False, False], index=[1, 2, 3])

print(x)

x.astype(int)

print(x)

x.astype(bool)

print(x)


