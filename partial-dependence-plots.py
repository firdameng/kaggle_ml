import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer

cols_to_use = ['Distance', 'Landsize', 'BuildingArea']


def get_some_data():
    data = pd.read_csv('./data/melb_data.csv')
    y = data.Price
    X = data[cols_to_use]
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y


X, y = get_some_data()
my_model = GradientBoostingRegressor()
my_model.fit(X, y)
my_plots = plot_partial_dependence(my_model,
                                   features=[0, 2],
                                   X=X,
                                   feature_names=cols_to_use,
                                   grid_resolution=10)

def get_some_data():
    cols_to_use = ['Distance', 'Landsize', 'BuildingArea']
    data = pd.read_csv('./data/melb_data.csv')
    y = data.Price
    X = data[cols_to_use]
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

# get_some_data is defined in hidden cell above.
X, y = get_some_data()
# scikit-learn originally implemented partial dependence plots only for Gradient Boosting models
# this was due to an implementation detail, and a future release will support all model types.
my_model = GradientBoostingRegressor()
# fit the model as usual
my_model.fit(X, y)
# Here we make the plot
# artial dependence plots show how each variable or predictor affects the model's predictions
# If you are familiar with linear or logistic regression models, partial dependence plots can be interepreted similarly to the coefficients in those models.
# feature_names 都要给出来
my_plots = plot_partial_dependence(my_model,
                                   features=[0,1, 2], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names=['Distance', 'Landsize', 'BuildingArea'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis

titanic_data = pd.read_csv('./data/titanic/train.csv')
titanic_y = titanic_data.Survived
clf = GradientBoostingClassifier()
titanic_X_colns = ['PassengerId','Age', 'Fare',]
titanic_X = titanic_data[titanic_X_colns]
my_imputer = Imputer()
imputed_titanic_X = my_imputer.fit_transform(titanic_X)

clf.fit(imputed_titanic_X, titanic_y)
titanic_plots = plot_partial_dependence(clf, features=[1,2], X=imputed_titanic_X,
                                        feature_names=titanic_X_colns, grid_resolution=8)
print('hh')