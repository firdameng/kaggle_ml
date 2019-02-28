import pandas as pd

data = pd.read_csv('./data/AER_credit_card_data.csv',
                   true_values = ['yes'],
                   false_values = ['no'])
print(data.head())

data.shape

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

y = data.card
X = data.drop(['card'], axis=1)

# Since there was no preprocessing, we didn't need a pipeline here. Used anyway as best practice
modeling_pipeline = make_pipeline(RandomForestClassifier())
# estimator : estimator object implementing 'fit'
#         The object to use to fit the data.
cv_scores = cross_val_score(modeling_pipeline, X, y, scoring='accuracy')  # 做分类的
print("Cross-val accuracy: %f" %cv_scores.mean())

expenditures_cardholders = data.expenditure[data.card]
expenditures_noncardholders = data.expenditure[~data.card]

print('Fraction of those who received a card with no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))
print('Fraction of those who received a card with no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))

# 移除这些相关性非常强的列
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)
cv_scores = cross_val_score(modeling_pipeline, X2, y, scoring='accuracy')
print("Cross-val accuracy: %f" %cv_scores.mean())