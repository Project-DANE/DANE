import pandas as pd
import numpy as np 

import 

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
                       'dIncome': 'monthly_inc',
                       'TotalIncome': 'total_inc',
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

