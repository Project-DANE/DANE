# imports

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats


def vis_countplot(train, col = 'GuarantorRequired'):
    ''' takes in a column name and a dataframe and show countplot graph'''
    
    #plot countplot graph
    sns.countplot(x=col, hue='bad_resident', data=train)
    sns.despine()
    plt.title('Gurrantor Required relationship with Resident Type')
    plt.xlabel('Gurrantor Required')
    plt.show()
    
    
def chi_test_d(train, col = 'GuarantorRequired'):
    '''takes in a column name and a dataframe and runs chi-square test to compare relationship of bad_resident 
    with a datframe attributes 
    '''
    
    # set alpha value to 0.05
    alpha = 0.05
    
    # set null and alternative hypothesis 
    null_hypothesis = col + ' and bad_resident are independent'
    alternative_hypothesis = col + ' and bad_resident are dependent'

    # create an observed crosstab, or contingency table from a dataframe's two columns
    observed = pd.crosstab(train[col], train.bad_resident)

    # run chi-square test
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    # print Null Hypothesis followed by a new line
    print(f'Null Hypothesis: {null_hypothesis}\n')

    # print Alternative Hypothesis followed by a new line
    print(f'Alternative Hypothesis: {alternative_hypothesis}\n')

    # print the chi2 value
    print(f'chi^2 = {chi2}') 

    # print the p-value followed by a new line
    print(f'p     = {p}\n')

    if p < alpha:
        print(f'We reject null hypothesis')
        print(f'There exists some relationship between {col} and bad_resident.')
    else:
        print(f'We fail to reject null hypothesis')
        print(f'There appears to be no significant relationship between {col} and bad_resident.')
        

def viz_rent(train):
    '''plot histogram'''
    
    bins = [829,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,4000,5000,6000]
    
    plt.figure(figsize=(8, 16))
    
    plt.subplot(211)
    rent_bin = pd.cut(train['rent'], bins = bins)
    sns.countplot(y=rent_bin,hue='bad_resident',data=train)
    sns.despine()
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.title('Relation of rent with Resident')
    
    plt.subplot(212)
    train_bad_resident = train[train['bad_resident']==1]
    rent_bin_bad = pd.cut(train_bad_resident['rent'], bins = bins)
    sns.countplot(y=rent_bin_bad,data=train_bad_resident)
    sns.despine()
    plt.title('Relation of rent with Bad Resident')
    
    
def chi_test(bins, train):
    '''get result of chi-square for a feature to churn'''

    observed = pd.crosstab(bins, train.bad_resident)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    #print(f'chi² = {chi2:.3f}')
    #print(f'p = {p:.3}')
    return chi2, p


def countplot(data, column, color):
    
    sns.countplot(x = column, data = data, color = color, ec = 'black')
    plt.title(f'NUMBER OF RESIDENTS BY {column.upper()}')
    plt.xlabel(f'{column.capitalize()}')
    plt.ylabel('Count')
    plt.show()    
    