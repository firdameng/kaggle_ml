import pandas as pd
import numpy as np
from pandas import Series,DataFrame

data_train = pd.read_csv("./data/titanic/train.csv")
# print(data_train)
# print(data_train.info())
print(data_train.describe())