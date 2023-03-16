import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean(currency):
    '''Takes a currency considered an obj or str and turns it into a clean float
        rounded to two decimal places'''
    currency = currency.replace('$', '')
    currency = currency.replace(',', '')
    currency = currency.replace('-', '')
    currency = round(float(currency), 2)
    return currency


def train_vailidate_test_split(df, target, strat = None):
    '''
    splits the data inserted into a train test validate split
    if you are going to stratify you must give a third argument
    if you are not going to stratify only use two arguments
    '''
    if strat:
        train_validate, test = train_test_split(df, train_size =.8, random_state = 91, stratify = df[target])
        train, validate = train_test_split(train_validate, train_size = .7, random_state = 91, stratify = train_validate[target])
    else:
        train_validate, test = train_test_split(df, train_size =.8, random_state = 91)
        train, validate = train_test_split(train_validate, train_size = .7, random_state = 91)
    X_train = train.drop(columns=target)
    y_train = train[target]
    X_val = validate.drop(columns=target)
    y_val = validate[target]
    X_test = test.drop(columns=target)
    y_test = test[target]
    return train, validate, test, X_train, y_train, X_val, y_val, X_test, y_test


def scale_splits(X_train, X_val, X_test, scaler, columns = False):
    '''
    Accepts input of a train validate test split and a specific scaler. The function will then scale
    the data according to the scaler used and output the splits as scaled splits
    If you want to scale by specific columns enter them in brackets and quotations after entering scaler
    otherwise the function will scale the entire dataframe
    '''
    if columns:
        scale = scaler.fit(X_train[columns])
        train_initial = pd.DataFrame(scale.transform(X_train[columns]),
        columns= X_train[columns].columns.values).set_index([X_train.index.values])
        val_initial = pd.DataFrame(scale.transform(X_val[columns]),
        columns= X_val[columns].columns.values).set_index([X_val.index.values])
        test_initial = pd.DataFrame(scale.transform(X_test[columns]),
        columns= X_test[columns].columns.values).set_index([X_test.index.values])
        train_scaled = X_train.copy()
        val_scaled = X_val.copy()
        test_scaled = X_test.copy()
        train_scaled.update(train_initial)
        val_scaled.update(val_initial)
        test_scaled.update(test_initial)

    else:
        scale = scaler.fit(X_train)
        train_scaled = pd.DataFrame(scale.transform(X_train),
        columns= X_train.columns.values).set_index([X_train.index.values])
        val_scaled = pd.DataFrame(scale.transform(X_val),
        columns= X_val.columns.values).set_index([X_val.index.values])
        test_scaled = pd.DataFrame(scale.transform(X_test),
        columns= X_test.columns.values).set_index([X_test.index.values])
        train_scaled = X_train.copy()
        val_scaled = X_val.copy()
        test_scaled = X_test.copy()
    return train_scaled, val_scaled, test_scaled

def remove_outliers(df, k=1.5, columns = False):
    '''
    This function is to remove the top 25% and bottom 25% of the data for each column.
    This removes the top and bottom 50% for every column to ensure all outliers are gone.
    '''
    a=[]
    b=[]
    fences=[a, b]
    features= []
    col_list = []
    i=0
    for col in df:
            new_df=np.where(df[col].nunique()>8, True, False)
            if new_df==True:
                if df[col].dtype == 'float' or df[col].dtype == 'int':
                    '''
                    for each feature find the first and third quartile
                    '''
                    q1, q3 = df[col].quantile([.25, .75])
                    '''
                    calculate inter quartile range
                    '''
                    iqr = q3 - q1
                    '''
                    calculate the upper and lower fence
                    '''
                    upper_fence = q3 + (k * iqr)
                    lower_fence = q1 - (k * iqr)
                    '''
                    appending the upper and lower fences to lists
                    '''
                    a.append(upper_fence)
                    b.append(lower_fence)
                    '''
                    appending the feature names to a list
                    '''
                    features.append(col)
                    '''
                    assigning the fences and feature names to a dataframe
                    '''
                    var_fences= pd.DataFrame(fences, columns=features, index=['upper_fence', 'lower_fence'])
                    col_list.append(col)
                else:
                    print(col)
                    print('column is not a float or int')
            else:
                print(f'{col} column ignored')
    '''
    for loop used to remove the data deemed unecessary
    '''
    for col in col_list:
        df = df[(df[col]<= a[i]) & (df[col]>= b[i])]
        i+=1
    return df, var_fences