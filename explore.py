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
    plt.legend(labels= ['Good Resident','Bad resident'])
    plt.show()
    
    
def chi_test_g(train, col = 'GuarantorRequired'):
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
        

def viz_rent(train, col):
    '''plot histogram'''
    
    bins = [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500, 2600]
    
    plt.figure(figsize=(8, 16))
    
    plt.subplot(211)
    rent_bin = pd.cut(train[col], bins = bins)
    sns.countplot(y=rent_bin,hue='bad_resident',data=train)
    sns.despine()
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.title('Relation of rent with Resident')
    plt.xlabel('Count')
    plt.ylabel(col.capitalize())
    plt.legend(labels= ['Good Resident','Bad resident'])
    
    plt.subplot(212)
    train_bad_resident = train[train['bad_resident']==1]
    rent_bin_bad = pd.cut(train_bad_resident['rent'], bins = bins)
    sns.countplot(y=rent_bin_bad,data=train_bad_resident, color = 'seagreen')
    sns.despine()
    plt.title('Relation of rent with Bad Resident')
    plt.xlabel('Count')
    plt.ylabel(col.capitalize())
    
    
    
def chi_test(train, col = False, bins = False):
    '''get result of chi-square test'''
    
    if bins:
    
        binss = [0, 5, 10, 15, 20, 25, 30, 35, 40 , 45, 50, 55, 60, 65, 70, 75, 80]
        age_bin = pd.cut(train.age, bins= binss)
        
        observed = pd.crosstab(age_bin, train.bad_resident)
        chi2, p, degf, expected = stats.chi2_contingency(observed)
    else:
        
        observed = pd.crosstab(train[col], train.bad_resident)
        chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    𝜶 = .05

    if p < 𝜶:
        print("We reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")
    
    return print(f'''
Chi2 = {chi2:.3f}
P-value = {p:.3f}''')



def countplot(data, column, color, bad = False):
    
    
    if bad: 
        bad_resid = data[(data.bad_resident == 1)]
        sns.countplot(x = column, data = bad_resid, color = color, ec = 'black')
        plt.title(f'NUMBER OF RESIDENTS BY {column.upper()}')
        plt.xlabel(f'{column.capitalize()}')
        plt.ylabel('Count')
        plt.show()
    
    else:
        
        sns.countplot(x = column, data = data, color = color, ec = 'black')
        plt.title(f'NUMBER OF RESIDENTS BY {column.upper()}')
        plt.xlabel(f'{column.capitalize()}')
        plt.ylabel('Count')
        plt.show()
        
    
def histplot_n(data, col, bad = False):
    
    
    bins = [18, 20, 25, 30, 35, 40 , 45, 50, 55, 60, 65, 70, 75, 80]
    
    if bad:
    
        bad_resid = data[(data.bad_resident == 1)]
    
        sns.histplot(data= bad_resid, x= col, bins=bins, color = 'seagreen') 
        
        plt.title(f'{col.capitalize()} Causing The Most Damage')
        plt.xlabel(f'{col.capitalize()}')
    
    else:
        
        sns.histplot(data= data, x= col, bins= bins, hue= "bad_resident", multiple="stack")
    
        plt.title(f'{col.capitalize()} Causing The Most Damage')
        plt.xlabel(f'{col.capitalize()}')
    
        plt.legend(labels= ['Bad Resident','Good resident'])
    
    return plt.show()
    

 
    