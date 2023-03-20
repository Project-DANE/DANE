# imports

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats



def get_common(df1):
    
    '''
    This functions filters out the negative charge codes, then gets the top six of those codes.
    It then returns a plot to show the results. 
    '''
    
    # negative charge codes
    cc = [96, 105, 106, 115, 127, 131, 137, 138,142, 148, 154,
                155, 156, 163, 166, 169, 183, 189, 192, 229, 231, 233,
                245, 246, 247, 248, 249,250, 251, 253, 268, 298]
    
    
    # top six negative charge codes
    six= [298, 155, 105, 154, 156, 131]
    
    order= [298, 105, 155, 154, 156, 131]
    
    # create new df using negative charge codes
    bad_df= df1[df1['charge_code'].isin(cc)]
    
    # create new df using the top six negative charge codes
    six_df= bad_df[bad_df['charge_code'].isin(six)]
    
    #plotting the results of the function
    color= ['red', 'grey', 'grey', 'grey', 'grey', 'grey',]
    ax = sns.countplot(
                     data= six_df , x= 'charge_code', palette= color,
                     order= order)
    
    # Set xlabel
    plt.xlabel('Total Count')
    
    # Set ylabel
    plt.ylabel('Charge Code')
    
    # Set plot title
    plt.title('Total Charge Code Count')
    
    # set font scale
    sns.set(font_scale= 5)
    
    # Show plot
    plt.show()
    
    
def bad_properties(train):
    """
    This function takes in the train dataset and returns the 10 properties with 
    the highest number of charge codes
    """
    
    # bad charge codes
    cc = [96, 105, 106, 115, 127, 131, 137, 138,142, 148, 154,
                155, 156, 163, 166, 169, 183, 189, 192, 229, 231, 233,
                245, 246, 247, 248, 249,250, 251, 253, 268, 298]
    
    
    six= [298, 105, 155, 154, 156, 131]
    
    # filter out the charge codes
    bad_properties= train[train['charge_code'].isin(cc)]
    
    # return the properties with the most charge codes
    df3= bad_properties.groupby('prop_id')['charge_code'].count().nlargest(10)
    
    # turn results into a dataframe
    df3= pd.DataFrame({'most_common': df3})
    
    # reset index
    df3= df3.reset_index()
    return df3    


def plot_bad_properties(df3):
    '''
    This function plots the results of the top 10 properties with the 
    highest count of charge codes
    '''
    
    # sets the graph color
    color= ['red', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey']
    fig = plt.figure()
    # create the graphs in seaborn
    ax= sns.barplot(data= df3, x= 'prop_id', y= 'most_common', palette= color, ec= 'black' , linewidth= 3.8)
    sns.set(rc={'figure.figsize':(39.7,18.27)})
    sns.set(font_scale= 4.5)
    plt.xlabel('Property ID')
    plt.ylabel('Charge Code Count')
    plt.title('Properties With The Most Damage Codes')
    for i in ax.containers:
            ax.bar_label(i,)

def vis_countplot(train, col = 'GuarantorRequired'):
    ''' takes in a column name and a dataframe and show countplot graph'''
    
    #plot countplot graph
    sns.countplot(x=col, hue='bad_resident', data=train)
    sns.despine()
    plt.title('Gurrantor Required relationship with Resident Type')
    plt.xlabel('Gurrantor Required')
    plt.show()
    
    
def chi_test(train, col = 'GuarantorRequired'):
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
    