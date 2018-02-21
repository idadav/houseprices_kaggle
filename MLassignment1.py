#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:50:06 2018

@author: idadav
"""

# import libraries and dataset
import pandas as pd
import numpy as np

df = pd.read_csv('~/Documents/Programming/Kaggle/House Prices/train.csv')
test = pd.read_csv('~/Documents/Programming/Kaggle/House Prices/test.csv')

# first overlook
df.head()
test.head()

# delete Id column as not necessary 
del df['Id']

# take out Id from test data for later
test_id = test['Id']
del test['Id']

"""
Check for outliers
"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(x = df['GrLivArea'], y = df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# remove the two large outliers
df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)
df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']>700000)].index)

"""
Dealing with NA's
"""

# check for NA's
print(df.isnull().sum())

# there seem to be a lot of NA's. We need to find them and deal with them.

# the NA's below simply represent that the house does not have that feature.
for col in ('Alley', 'BsmtQual', 'FireplaceQu', 'GarageType', 'GarageQual', 
            'GarageCond', 'PoolQC', 'MiscFeature', 'Fence', 'GarageFinish', 
            'BsmtExposure', 'BsmtCond', 'BsmtFinType1', 'MasVnrType', 
            'BsmtFinType2', 'MSSubClass'):
    df[col] = df[col].fillna('No')    

# filling Electrical with the mode
df['Electrical'] = df['Electrical'].fillna(test['Electrical'].mode()[0])

# imputing 0 for MasVnrArea and GarageYrBlt
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0) 

# median value for lot frontage (assuming people have similar lot frontage in the area)
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

# dropping Utilities due to a very high number of NA's
df['Utilities'] = df.drop(['Utilities'], axis=1)

print(df.isnull().sum().sort_values())

# same imputation on test set

# the NA's below simply represent that the house does not have that feature.
for col in ('Alley', 'BsmtQual', 'BsmtExposure', 'BsmtCond', 'BsmtFinType1', 
           'BsmtFinType2', 'Fence', 'FireplaceQu', 'GarageType', 'GarageQual',
           'GarageCond', 'GarageFinish', 'MasVnrType', 'MiscFeature', 
           'PoolQC'):
    test[col] = test[col].fillna('No')
 
    # same for the below but imputing with 0
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
            'BsmtFullBath', 'BsmtHalfBath', 'BedroomAbvGr', 'LotFrontage', 
            'MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars'):
    test[col] = test[col].fillna(0)

# median value for lot frontage (assuming people have similar lot frontage in the area)    
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

# mode value for the below
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
 
# dropping Utilities   
test['Utilities'] = test.drop(['Utilities'], axis=1)

# filling Functional with 'Typ'
test['Functional'] = test['Functional'].fillna('Typ')

print(test.isnull().sum().sort_values())

"""
Transform the data
"""

# 3 numeric variables are factor on train and test set
df['MSSubClass'] = df['MSSubClass'].astype('category')
df['MoSold'] = df['MoSold'].astype('category')
df['YrSold'] = df['YrSold'].astype('category')

test['MSSubClass'] = test['MSSubClass'].astype('category')
test['MoSold'] = test['MoSold'].astype('category')
test['YrSold'] = test['YrSold'].astype('category')

# adding total sqr foot of a house
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

# imputing the mode in case of NA's
test['TotalSF'] = test['TotalSF'].fillna(test['TotalSF'].mode()[0])

"""
Colinearity in train variables
"""
import seaborn as sns

correl = df.corr()
sns.heatmap(correl, mask=np.zeros_like(correl, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)

"""
Skewness of variables
"""

# check skewness and kurtosis in target varaible
from scipy.stats import kurtosis, skew

plt.style.use('ggplot')

plt.hist(df['SalePrice'], bins='auto')
print('mean : ', np.mean(df['SalePrice']))
print('variance : ', np.var(df['SalePrice']))
print('skewness : ', skew(df['SalePrice'])) # data is skewed
print('kurtosis : ', kurtosis(df['SalePrice']))

# log of target variable to normalise skewness
df['SalePrice'] = np.log1p(df['SalePrice'])
plt.hist(df['SalePrice'], bins='auto')

# combine data for log transformation and remove target variable
ntrain = df.shape[0]
ntest = test.shape[0]
y_train = df.SalePrice.values
all_data = pd.concat((df, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

# checking skewness for the rest of numeric data and transforming 
numeric_vars = all_data.dtypes[all_data.dtypes != 'object'].index

skewed_vars = all_data[numeric_vars].apply(lambda x: skew(x.dropna()))
skewed_vars = skewed_vars[skewed_vars > 0.75]
skewed_vars = skewed_vars.index
all_data[skewed_vars] = np.log1p(all_data[skewed_vars])

## TEST to see if improves RMSE: remove variables from dataset that had >1000 NAs originally
# it did improve RMSE so will keep this
all_data = all_data.drop('Alley', 1)
all_data = all_data.drop('Fence', 1)
all_data = all_data.drop('MiscFeature', 1)
all_data = all_data.drop('PoolQC', 1)
all_data = all_data.drop('FireplaceQu', 1)

"""
Converting to dummies for modelling and splitting back to train and test
"""

# getting dummies
all_data = pd.get_dummies(all_data)

# splitting back into train and test
x_df = all_data[:ntrain]
x_test = all_data[ntrain:]

"""
LassoCV Regression 
"""
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

# mean squared error function for after I fit my model
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, x_df, y_train, 
                                    scoring='neg_mean_squared_error', cv = 5))
    return(rmse)

# Lasso regression with cross-validation first
lasso_m = make_pipeline(RobustScaler(), 
                        LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(x_df, y_train))

# checking RMSE
rmse_cv(lasso_m).mean()

# prediction
lasso_preds = np.expm1(lasso_m.steps[1][1].predict(x_test))

# convert to CSV for kaggle
sol_lasso = pd.DataFrame({'id':test_id, 'SalePrice':lasso_preds})
sol_lasso.to_csv('MLsol4.csv', index = False)


"""
Elastic Net 
"""
from sklearn.linear_model import ElasticNetCV, ElasticNet

en_m = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio = [0.5], 
                    alphas = [1, 0.1, 0.001, 0.0005])).fit(x_df, y_train)

# check RMSE
rmse_cv(en_m).mean() # not better than lasso

# predict on test data
elasticnet_predictions = np.expm1(en_m.steps[1][1].predict(x_test))

# convert to CSV for Kaggle
sol_en = pd.DataFrame({'id':test_id, 'SalePrice':elasticnet_predictions})
sol_en.to_csv('MLsol_en.csv', index = False)

"""
Ridge
"""
from sklearn.linear_model import RidgeCV, Ridge

# fit the ridge model
ridge_m = make_pipeline(RobustScaler(), 
                        RidgeCV(alphas = [1, 0.1, 0.001, 0.0005])).fit(x_df, y_train)

# check RMSE
rmse_cv(ridge_m).mean() # much worse result than lasso or elastic net

"""
Trying out some simple stacking to see if I can improve my score
"""
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

# models
lso = Lasso(random_state=123)
rdge = Ridge(random_state=123)
lr = LinearRegression()
regressors = [rdge, lr]
stacks = StackingRegressor(regressors = regressors,
                           meta_regressor = lso)

elnet = ElasticNet(random_state=123)

parameters = {'ridge__alpha': [0.1, 0.001, 0.0005], 
                'lasso__alpha': [0.1, 0.001, 0.0005]
              }

grd_s = GridSearchCV(estimator=stacks,
                     param_grid=parameters,
                     cv=5,
                     refit=True)

grd_s.fit(x_df, y_train)

"""
GridSearchCV with LassoCV (can't get the fit to work for some reason)
"""

lasso = LassoCV()
lasso_alphas = {'alphas': [1, 0.1, 0.001, 0.0005]}

lasso_gridsearch = GridSearchCV(lasso, lasso_alphas, cv=5, 
                                scoring='neg_mean_absolute_error')

lasso_gridsearch.fit(x_df, y_train)

# check RMSE
rmse_cv(lasso_gridsearch).mean() # seems decent!

# make predictions
lasso_grid_preds = np.expm1(lasso_gridsearch.predict(x_test))

# convert to CSV for Kaggle
sol_lasso_grid = pd.DataFrame({'id':test_id, 'SalePrice':lasso_grid_preds})
sol_lasso_grid.to_csv('MLlasso_g_1.csv', index = False)
