import pandas as pd
data = pd.read_csv('./data/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price

# Given these tradeoffs, when should you use each approach? On small datasets,
# the extra computational burden of running cross-validation isn't a big deal.
# These are also the problems where model quality scores would be least reliable with train-test split.
# So, if your dataset is smaller, you should run cross-validation.

# For the same reasons, a simple train-test split is sufficient for larger datasets.
# It will run faster, and you may have enough data that there's little need to re-use some of it for holdout.

# There's no simple threshold for what constitutes a large vs small dataset.
# If your model takes a couple minute or less to run, it's probably worth switching to cross-validation.
# If your model takes much longer to run, cross-validation may slow down your workflow more than it's worth.
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

from sklearn.model_selection import cross_val_score
# None, to use the default 3-fold cross validation,
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

