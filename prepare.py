import pandas as pd
import numpy as np

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



def prep(df):

    '''takes a dataframe, drops repetative columns, filters for current status, resets index and returns
    a dataframe'''
    
    # Drop repetative columns that contained same inofrmation
    df = df.drop(columns = ['HMY', 'HMYPerson', 'Rent', 'SRENT', 'hTenant', 'hTent', 'HMY1'])
    
    # Rename columns into a pythonic format
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

    # Eliminate duplicate charges by citing only current leases
    df = df[df.sStatus == 'Current']
    
    # Reset the index to account for duplicates dropped
    df = df.reset_index(drop=True)
    
    # return a dataframe
    return df