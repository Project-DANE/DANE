import pandas as pd
import numpy as np
from scipy import stats


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