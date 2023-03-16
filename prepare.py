import pandas as pd
import numpy as np

<<<<<<< HEAD
def acquire():
    df = pd.read_csv('cws_residents.csv')
    # Pull CSV that has been queried from CWS database
    return df

=======
>>>>>>> 57891566af01dbf74ca194beba4034fa9dc87933

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
                   'HProperty': 'property_id',
                   'sStatus': 'status',
                   'SNAME':'charge_name',
                   'cRent': 'rent',
                   'iTerm': 'term',
                   'dIncome': 'monthly_income',
                   'GuarantorRequired': 'guarrantor_required',
                   'TotalIncome': 'total_income',
                   'Recommendation': 'recommendation',
                   'AverageApplicantAge': 'age',
                   'AvgRiskScore':'risk_score',
                   'ReasonThatDroveDecisionDescription': 'reason'})

    # Eliminate duplicate charges by citing only current leases
    df = df[df.status == 'Current']
    
    # Reset the index to account for duplicates dropped
    df = df.reset_index(drop=True)
    
    # return a dataframe
    return df