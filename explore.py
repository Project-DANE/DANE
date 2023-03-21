# imports

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats


def get_common(train):
    
    '''
    This functions filters out the negative charge codes, then gets the top six of those codes.
    It then returns a plot to show the results. 
    '''
    
    # negative charge codes
    cc = [96, 105, 106, 115, 127, 131, 137, 138,142, 148, 154,
                155, 156, 163, 166, 169, 183, 189, 192, 229, 231, 233,
                245, 246, 247, 248, 249,250, 251, 253, 268, 298]
    
    
    # top six negative charge codes
    six= [298, 105, 155, 154, 156, 131]
    
    # create new df using negative charge codes
    bad_df= train[train['charge_code'].isin(cc)]
    
    # create new df using the top six negative charge codes
    six_df= bad_df[bad_df['charge_code'].isin(six)]
    
    #plotting the results of the function
    color= ['grey', 'grey', 'grey', 'grey', 'grey','red']
    bar = sns.countplot(data= six_df , x= 'charge_code', color = 'grey', ec= 'black')
    patch_h = [patch.get_height() for patch in bar.patches]   
    idx_tallest = np.argmax(patch_h)   
    bar.patches[idx_tallest].set_facecolor('seagreen')
    plt.title('Most Common Charge Codes')
    plt.xlabel('Charge')
    plt.ylabel('Count')
    for i in bar.containers:
            bar.bar_label(i,)
            
    return plt.show()        
          
def bad_properties(train):
    
    '''
    This function returns a dataframe that only contains residents with
    negative charge codes.
    '''
    
    # negative charge codes
    cc = [96, 105, 106, 115, 127, 131, 137, 138,142, 148, 154,
                155, 156, 163, 166, 169, 183, 189, 192, 229, 231, 233,
                245, 246, 247, 248, 249,250, 251, 253, 268, 298]
    
    # most common negative charge codes
    six= [298, 105, 155, 154, 156, 131]
    
    # filters our the bad properties
    bad_properties= train[train['charge_code'].isin(cc)]
    
    # returning the most 10 most common negative charge codes 
    df3= bad_properties.groupby('prop_id')['charge_code'].count().nlargest(10)
    
    
    # creates a dataframe of the results 
    df3= pd.DataFrame({'most_common': df3})
    
    # rest the index of the dataframe
    df3= df3.reset_index()
    
    # return the dataframe
    return df3
    
    
def plot_bad_properties(train):
    
    '''
    This function returns a visual of the states with the highest
    amount of charge codes.
    '''
    
    # setting the palette order 
    cl= ['red', 'grey', 'grey', 'grey', 'grey']
    
    # returing the resulting dataframe from the `bad_properties` function
    df = bad_properties(train)
    values = np.array(train.prop_id)
    fig = plt.figure()
    
    # creating the graph
    bar = sns.barplot(data= df, x= 'prop_id', y= 'most_common', palette= cl,  errwidth=0, ec= 'black')
    patch_h = [patch.get_height() for patch in bar.patches]   
    idx_tallest = np.argmax(patch_h)   
    
    # setting the xlabel
    plt.xlabel('Property ID')
    
    # setting the ylabel
    plt.ylabel('Charge Code Count')
    
    # setting the title
    plt.title('Properties With The Most Damage Codes')
    
    # adding the total count number on the top of the bars
    for i in bar.containers:
            bar.bar_label(i,)
    
    # returning the graph
    return plt.show()
            
def risk_score(train): 
    
    """
    This function shows how many bad residetns are in each range of risk
    score.
    """
    one= 1
    # set the color palette order
    color= ['grey', 'grey', 'grey', 'red', 'red']
    
    # set the font scale
    sns.set(font_scale= one)
    
    # Creating custom bins with 100 range and binning both total train pop and only bad train pop
    bins = [300,400,500,600,700,800,900]
    risk_bin = pd.cut(train['risk_score'], bins = bins)
    bad_risk = pd.cut(train[train.bad_resident == 1].risk_score, bins = bins)
    
    # creating the graph 
    ax= sns.countplot(x = bad_risk, palette= color, ec= 'black')
    
    # set xlabel
    plt.xlabel('Risk Score Range')
    
    # set ylabel
    plt.ylabel('Total Count')
    
    # set graph title
    plt.title('Which Range of Risk Score Has The Most Bad Residents')
    
    # set the font scale for the graph
    sns.set(font_scale= 1.5)
    
    # adds the count number to the top of the bars
    for i in ax.containers:
            ax.bar_label(i,)
    return plt.show()

def countplot_n(data, column, bad=False):
    
    """
    This function retruns the percentage of bad residents 
    per lease term length.
    """
    
    color= ['grey', 'grey', 'grey', 'grey', 'red', 'red', 'grey', 'red', 'grey', 'grey']
    
     # set the font scale
    sns.set(font_scale= 1)
    
    # if statement for which graph will be returned 
    if bad:
        bad_resid = data[data.bad_resident == 1]
        bar= sns.countplot(x= column, data=bad_resid, palette= color, ec='black')
        plt.title(f'Number of Bad Residents by {column.capitalize()}')
        plt.xlabel(f'{column.capitalize()} Length')
        plt.ylabel('Total Count')
        for p in bar.patches:
            height= p.get_height() / len(bad_resid) * 100
            bar.annotate(f"{height:.2f}%", (p.get_x() + p.get_width() / 2,
                                            p.get_height()), ha='center', va='center',
                                            xytext=(0, 5), textcoords='offset points')
        plt.show()
    
    else:
        bar = sns.countplot(x=column, data=data, palette= color, ec= 'black')
        plt.title(f'Number of Bad Residents by {column.capitalize()}')
        plt.xlabel(f'{column.capitalize()} Length')
        plt.ylabel('Total Count')
        for p in bar.patches:
            height = p.get_height() / len(data) * 100
            bar.annotate(f"{height:.2f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha= 'center', va= 'center', xytext= (0, 5), textcoords= 'offset points')
        plt.show()
    


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
    

 
    