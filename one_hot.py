# Read the data
import pandas as pd
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# Drop houses where the target is missing
# 0, or 'index' : Drop rows which contain missing values.
# 在saleprice这一列查看缺失值，
# inplace参数的理解：
# 修改一个对象时：
#  inplace=True：不创建新的对象，直接对原始对象进行修改；
#  inplace=False：对数据进行修改，创建并返回新的对象承载其修改结果。
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

target = train_data.SalePrice

# Since missing values isn't the focus of this tutorial, we use the simplest
# possible approach, which drops these columns.
# For more detail (and a better approach) to missing values, see
# https://www.kaggle.com/dansbecker/handling-missing-values
cols_with_missing = [col for col in train_data.columns
                                 if train_data[col].isnull().any()]

# [id saleprice missing_value_columns]被删除
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)

# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
# 统计某一列不重复数值的个数，用它来大致判断某一列为目录列
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols

# 既含有目录列，也含有数值列
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

train_predictors.dtypes.sample(10)    # 返回该类型的随机样本，起查看的效果在gupyter notebook
# 这样一高，数据就很稀疏了，但效果很好呀，相比直接剔除
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X, y,
                                scoring = 'neg_mean_absolute_error').mean()

# 排除含对象的列
predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))

#if the training dataset and test datasets get misaligned, your results will be nonsense.
# This could happen if a categorical had a different number of values in the training data vs the test data.

#Ensure the test data is encoded in the same manner as the training data with the align command:
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left',
                                                                    axis=1)