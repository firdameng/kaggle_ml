import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# Read Data
data = pd.read_csv('./data/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price
train_X, test_X, train_y, test_y = train_test_split(X, y)

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
# This particular pipeline was only a small improvement in code elegance.
# But pipelines become increasingly valuable as your data processing becomes increasingly sophisticated.
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)

# 性能上也提升了一丢丢
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

my_imputer = Imputer()
my_model = RandomForestRegressor()

imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)
my_model.fit(imputed_train_X, train_y)
predictions = my_model.predict(imputed_test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))