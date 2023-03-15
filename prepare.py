import python as pd
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
# Eliminate duplicate charges by 