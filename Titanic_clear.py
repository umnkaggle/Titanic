'''

Patrick Daly
UMN Kaggle
July 20, 2015

Titanic Data Set
- Machine Learning from Disaster

The ipython notebook (Titanic Workbook) is a huge collection of just about 
everything useful and should be used as a reference but can be 
overwhelming. This will be an example of a clean implementation. Generally 
ipython notebooks are meant for teaching and presentation. They can be a 
bit clunky for actual model building. Typical workflow seems to be split 
between a python script and an ipython shell. The shell serves as a 
sandbox for exploring the data and testing things out whereas the python 
script will be the current working copy. 

'''

import pandas as pd
from pandas import DataFrame
import numpy as np

import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

# Note: you can organize your code to read just like R where
#     the script is executed line by line top from top down.
#     I've split everything up into functions and a main so
#     the code is reusable and functions can be imported just
#     as a library like pandas or numpy.

def categorical_describe(df):
    return df[df.columns[df.dtypes == 'object']].describe()

def build_dummies(df):
    categoricals = list(df.columns[df.dtypes == 'object'])
    for variable in categoricals:
        dummies = pd.get_dummies(df[variable])
        df = pd.concat([df, dummies], axis=1)
        df.drop(variable, axis=1, inplace=True)
    return df

def _cv_score(model, df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    scores = []
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                train_size=0.8)
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    return np.mean(scores)

def _clean_up(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Family'] = df['Name'].apply(lambda name: name.split(',')[0]) +\
            df['FamilySize'].apply(lambda size: str(size))
    df.drop(['Ticket', 'Cabin', 'Embarked', 'Name'], axis=1,
            inplace=True)
    return _age_fill(df)

def _age_fill(df):
    df = build_dummies(df)
    df_age = df.dropna().copy()
    X = df_age.drop(['Age', 'Survived'], axis=1)
    y = df_age['Age']
    age_ols = sm.OLS(y,X).fit()
    df_age_missing = df[df['Age'].isnull()]
    df_age_missing.drop(['Age','Survived'], axis=1, inplace=True)
    df['Age'][df['Age'].isnull()] = age_ols.predict(df_age_missing)
    return df

def parameter_sweep(model, df, param_grid):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    clf = GridSearchCV(model, param_grid)
    clf.fit(X,y)
    return clf.grid_scores_

if __name__ == '__main__':

    # prep data
    train = pd.read_csv('train.csv', index_col='PassengerId')
    test  = pd.read_csv('test.csv', index_col='PassengerId')
    train_clean = _clean_up(train)

    # modeling
    param_grid = {'n_estimators': range(40, 160, 10)}
    p =  parameter_sweep(RandomForestClassifier(), train_clean, param_grid)
    p # for some reason when printing p the output isn't formatted
    rf = RandomForestClassifier(n_estimators=100)
    rf_scores = _cv_score(rf, train_clean)
    print rf_scores






