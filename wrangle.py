import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_cws_data(): 
    
    
    charge_codes = [96, 105, 106, 115, 127, 131, 137, 138,142, 148, 154,
                155, 156, 163, 166, 169, 183, 189, 192, 229, 231, 233,
                245, 246, 247, 248, 249,250, 251, 253, 268, 298]
    
    df = pd.read_csv('cws_residents.csv')
    # Pull CSV that has been queried from CWS database

    df = df.drop(columns = ['HMY', 'HMYPerson', 'Rent', 'SRENT', 'hTenant', 'hTent', 'HMY1'])
    # Drop repetative columns that contained same inofrmation

    df = df.rename(columns = {'HPerson': 'id',
                       'STOTALAMOUNT': 'total_charges',
                       'SAmountPaid': 'amount_paid',
                       'BOPEN': 'open',
                       'SNOTES': 'description',
                       'HRetentionacct': 'charge_code',
                       'HProperty': 'prop_id',
                       'SNAME':'charge_name',
                       'cRent': 'rent',
                       'iTerm': 'term',
                       'dIncome': 'monthly_income',
                       'TotalIncome': 'total_income',
                       'AverageApplicantAge': 'age',
                       'AvgRiskScore':'risk_score',
                       'ReasonThatDroveDecisionDescription': 'reason'})
    # Rename columns into a pythonic format

    df = df[df.sStatus == 'Current']
    # Eliminate duplicate charges by citing only current leases

    df = df.reset_index(drop=True)
    # Reset the index to account for duplicates dropped
    
    df['bad_resident'] = df['charge_code'].isin(charge_codes)

    df['bad_resident'] = np.where(df.bad_resident == True, 1, 0)
    
    df = df.drop_duplicates(subset = ['id', 'bad_resident'])
    df = remove_outliers(df, 'age')
    
    #Filter by bad resident
    df_bad = df[df['bad_resident'] == 1]

    # Get the indices of users who only have 0 status
    idx_0 = df[~df['id'].isin(df_bad['id'])].index

    # Filter rows based on 0 status for users who only have 0 status
    df_0 = df.loc[idx_0]

    # Combine the 1 and 0 dataframes
    df_combined = pd.concat([df_bad, df_0])

    # Sort by id to get the final result
    df = df_combined.sort_values('id')
    
    # Reset the index to account for duplicates dropped
    df = df.reset_index(drop=True)

    return df


def train_vailidate_test_split(df, target, strat = None):
    '''
    splits the data inserted into a train test validate split
    if you are going to stratify you must give a third argument
    if you are not going to stratify only use two arguments
    '''
    if strat:
        train_validate, test = train_test_split(df, train_size =.8, random_state = 91, stratify = df[target])
        train, validate = train_test_split(train_validate, train_size = .7,
                                           random_state = 91, stratify = train_validate[target])
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

def states(val):
    '''
    This funciton takes in a column of values and uses a previously established property key to 
    convert each property id into the name of the state in which the property resides
    '''
    if val in range(53,116) or val in range(152,159) or val in [198,218,229,252,440,441,442,458]:
        return 'Texas'
    elif val in range(116,124) or val in [159, 444]:
        return 'North Carolina'
    elif val in range(125,131) or val in [164,183,212,213,217, 253]:
        return 'Colorado'
    elif val in range(142,147) or val in [216]:
        return 'Arizona'
    elif val == 131:
        return 'California'
    elif val in range(132,142) or val in [385,443,459]:
        return 'Georgia'
    elif val in [277,280]:
        return 'Tennessee'
    elif val in range(147,152) or val in [160,161,162,163]:
        return 'Washington'

def remove_outliers(data, col):
    q1, q3 = np.percentile(data[col], [5, 95])
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_data = [x for x in data[col] if (x >= lower_bound) and (x <= upper_bound)]
    
    filtered_age = filtered_data
    filtered_df = data.loc[data[col].isin(filtered_age)]
    
    return filtered_df

def col_drop(df):
    df = df.drop(columns = ['id', 'total_charges', 'amount_paid', 'open', 'charge_code',
                             'description', 'charge_name', 'sStatus', 'reason', 'GuarantorRequired'])
    return df
        
def rent_change(val):
    if val in [0, 100]:
        return 1672
    else:
        return val
    
    
    
def model_prep(df):
    '''
    This model takes a complete df resplits it using the same random state along with dropping columns,
    aliasing columns, creating dummies and otherwise prepping the data for modeling
    '''
   
    df = remove_outliers(df, 'age')
    df = col_drop(df)
    df.monthly_income = np.where(df.monthly_income >= 20000, df.monthly_income/12, df.monthly_income)
    df.total_income = np.where(df.total_income == 0, df.monthly_income * 12, df.total_income)
    df.monthly_income = np.where(df.monthly_income == 0, df.total_income/12, df.monthly_income)
    df.total_income = np.where(df.total_income == 0, df.total_income.mean(), df.total_income)
    df.monthly_income = np.where(df.monthly_income == 0, df.monthly_income.mean(), df.monthly_income)
    df.prop_id = df.prop_id.apply(states)
    df.rent = df.rent.apply(rent_change)
    dummies = pd.get_dummies(df[['prop_id', 'Recommendation']])
    df = pd.concat([df, dummies], axis = 1)
    df = df.drop(columns = ['prop_id', 'Recommendation'])
    train, validate, test, X_train, y_train, X_val, y_val, X_test, y_test = train_vailidate_test_split(df, 'bad_resident','bad_resident')
    train_scaled, val_scaled, test_scaled = scale_splits(X_train, X_val, X_test, StandardScaler(),
                                                         columns = ['rent', 'monthly_income',
                                                                    'total_income', 'age', 'risk_score'])
    return train, validate, test, y_train, y_val, y_test, train_scaled, val_scaled, test_scaled
