import pandas as pd
import numpy as np
from scipy import stats


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
    

    return chi2, p


def countplot(data, column, color):
    
    sns.countplot(x = column, data = data, color = color, ec = 'black')
    plt.title(f'NUMBER OF RESIDENTS BY {column.upper()}')
    plt.xlabel(f'{column.capitalize()}')
    plt.ylabel('Count')
    plt.show()
    
    
def histplot_n(data, col):
    
    binss = [18, 20, 25, 30, 35, 40 , 45, 50, 55, 60, 65, 70, 75, 80]
    
    sns.histplot(data= data, x= col, bins= binss, hue= "bad_resident", multiple="stack")
    
    plt.title(f'{col.capitalize()} Causing The Most Damage')
    plt.xlabel(f'{col.capitalize()}')
    
    plt.legend(labels= ['Bad Resident','Good resident'])
    
    return plt.show()
    
    
def histplot_br(col):
    
    binss = [18, 20, 25, 30, 35, 40 , 45, 50, 55, 60, 65, 70, 75, 80]
    
    bad_resid = train[(train.bad_resident == 1)]
    
    sns.histplot(data= bad_resid, x= col, bins=binss, color = 'seagreen') 
    
    plt.title(f'{col.capitalize()} Causing The Most Damage')
    plt.xlabel(f'{col.capitalize()}')
    
    return plt.show()