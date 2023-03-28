import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats


def vis_countplot(train, col = 'GuarantorRequired'):
    ''' takes in a column name and a dataframe and show countplot graph'''
    
    plt.figure(figsize = (10,6))
    
    #plot countplot graph
    sns.countplot(x=col, hue='bad_resident', data=train, color = 'dodgerblue')
    sns.despine()
    plt.title('Guarantor Required relationship with Resident Type')
    plt.xlabel('Guarantor Required')
    plt.legend(labels= ['Good Resident','Bad Resident'])
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
        

def viz_rent(train):
    '''Plot precentage of bad resident within rent range'''

    r1 = (len(train[(train.rent >= 1300) & (train.rent < 1401) & (train.bad_resident == 1)]) /len(train[(train.rent >1300) & (train.rent < 1401)]))
    r2 = (len(train[(train.rent > 1400) & (train.rent < 1501) & (train.bad_resident == 1)]) /len(train[(train.rent > 1400) & (train.rent < 1501)]))
    r3 = (len(train[(train.rent > 1500) & (train.rent < 1601) & (train.bad_resident == 1)]) /len(train[(train.rent > 1500) & (train.rent < 1601)]))
    r4 = (len(train[(train.rent > 1600) & (train.rent < 1701) & (train.bad_resident == 1)]) /len(train[(train.rent >1600) & (train.rent < 1701)]))
    r5 = (len(train[(train.rent > 1700) & (train.rent < 1801) & (train.bad_resident == 1)]) /len(train[(train.rent >1700) & (train.rent < 1801)]))
    r6 = (len(train[(train.rent > 1800) & (train.rent < 1901) & (train.bad_resident == 1)]) /len(train[(train.rent >1800) & (train.rent < 1901)]))
    r7 = (len(train[(train.rent > 1900) & (train.rent < 6000) & (train.bad_resident == 1)]) /len(train[(train.rent >1900) & (train.rent < 6000)]))
    rs = pd.DataFrame(data = { 
                                '$1300-$1400': r1, '$1400-$1500': r2,'$1500-$1600': r3, '$1600-$1700': r4, 
                                '$1700-$1800': r5, '$1800-$1900': r6, '$2000 and above': r7,
                             }, index = [0])


    rs = rs.round(2) * 100



    plt.figure(figsize=(12, 6))
    color= ['#cccccc', '#cccccc' , '#cccccc', '#cccccc', 'red', '#cccccc', '#cccccc']
    ax = sns.barplot(data=rs, palette=color, edgecolor = ['#cccccc', '#cccccc' , '#cccccc', '#cccccc', 'black', '#cccccc', '#cccccc'])

    for p in ax.patches:
            ax.annotate(f"{round(p.get_height())}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize= 13)

    plt.title('Relation of Rent with Bad Resident',fontsize=14)
    ax.set_xlabel('Rent Range', fontsize=12)
    ax.set_ylabel('Percent', fontsize=12)
    plt.ylim(0,7)
    sns.despine()
    plt.show()
    
    
def chi_test(train, col = False, bins = False):
    '''get result of chi-square test'''
    
    if bins:
    
        binss = [0, 5, 10, 15, 20, 25, 30, 35, 40 , 45, 50, 55, 60, 65, 70, 75, 80]
        age_bin = pd.cut(train.age, bins= binss)
        
        observed = pd.crosstab(train.bad_resident, age_bin)
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


def countplot(data, column, color, bad=False):
    plt.figure(figsize=(10, 6))

    if bad:
        bad_resid = data[(data.bad_resident == 1)]
        sns.countplot(x=column, data=bad_resid, color=color, ec='black')
    else:
        sns.countplot(x=column, data=data, color=color, ec='black')

    plt.title(f'Number Of Residents By {column.capitalize()}')
    plt.xlabel(f'{column.capitalize()}')
    plt.ylabel('Count')

    ax = plt.gca()
    total = len(data)
    for p in ax.patches:
        height = p.get_height()
        pct = height / total * 100
        ax.text(p.get_x() + p.get_width() / 2, height, f'{pct:.1f}%', ha='center', va='bottom', rotation=30)

    plt.subplots_adjust(top=1.3)
    plt.show()
        
        
def histplot_n(data, col, bad = False):
    
    plt.figure(figsize = (10,6))
    
    bins = [18, 20, 25, 30, 35, 40 , 45, 50, 55, 60, 65, 70, 75, 80]
    
    if bad:
    
        bad_resid = data[(data.bad_resident == 1)]
    
        sns.histplot(data= bad_resid, x= col, bins=bins, color = 'dodgerblue', edgecolor = 'black') 
        
        plt.title(f'{col.capitalize()} Causing The Most Damage')
        plt.xlabel(f'{col.capitalize()}')
    
    else:
        
        sns.histplot(data= data, x= col, bins= bins, edgecolor = 'black',
                     hue= "bad_resident", palette = ['grey','dodgerblue'], multiple="stack")
    
        plt.title(f'{col.capitalize()} Causing The Most Damage')
        plt.xlabel(f'{col.capitalize()}')
    
        plt.legend(labels= ['Bad Resident','Good Resident'])
    
    return plt.show()
    
def bad_properties(train):
    
    '''
    This function returns a dataframe that only contains residents with
    negative charge codes.
    '''
    plt.figure(figsize = (10,6))
    
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
    cl= ['#E50000', '#cccccc', '#cccccc', '#cccccc', '#cccccc']
    
    # returing the resulting dataframe from the `bad_properties` function
    df = bad_properties(train)
    values = np.array(train.prop_id)
    plt.figure(figsize = (10,6))
    
    # creating the graph
    bar = sns.barplot(data= df, x= 'prop_id', y= 'most_common', palette= cl,  errwidth=0)
    patch_h = [patch.get_height() for patch in bar.patches]   
    idx_tallest = np.argmax(patch_h)   
    
    # setting the xlabel
    plt.xlabel('Property Location')
    
    # setting the ylabel
    plt.ylabel('Charge Code Count')
    
    # setting the title
    plt.title('Properties With The Most Damage Codes')
    
    # adding the total count number on the top of the bars
    for i in bar.containers:
            bar.bar_label(i)
    
    # returning the graph
    plt.show()
            

def get_common(train):
    
    '''
    This functions filters out the negative charge codes, then gets the top six of those codes.
    It then returns a plot to show the results. 
    '''
    plt.figure(figsize = (10,6))
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
    bar = sns.countplot(data= six_df , x= 'charge_code', color = '#cccccc')
    patch_h = [patch.get_height() for patch in bar.patches]   
    idx_tallest = np.argmax(patch_h)   
    bar.patches[idx_tallest].set_facecolor('#E50000')
    plt.title('Most Common Charge Codes')
    plt.xlabel('Charge')
    plt.ylabel('Count')
    for i in bar.containers:
            bar.bar_label(i,)
            
    return plt.show()        
         
    
def states(val):
    '''
    This funciton takes in a column of values and uses a previously established property key to 
    convert each property id into the name of the state in which the property resides
    '''
    if val in range(53,116) or val in range(152,159) or val in [198,218,229,252,440,441,442,458]:
        return 'Texas'
    elif val in range(116,124) or val in [159, 444]:
        return 'North Carolina'
    elif val in range(125,131) or val in [164,183,212,213,217, 253]:
        return 'Colorado'
    elif val in range(142,147) or val in [216]:
        return 'Arizona'
    elif val == 131:
        return 'California'
    elif val in range(132,142) or val in [385,443,459]:
        return 'Georgia'
    elif val in [277,280]:
        return 'Tennessee'
    elif val in range(147,152) or val in [160,161,162,163]:
        return 'Washington'    

    
def risk_score(train): 
    
    """
    This function shows how many bad residetns are in each range of risk
    score.
    """
    plt.figure(figsize = (10,6))
    one= 1
    # set the color palette order
    color= ['#cccccc', '#cccccc', '#cccccc', '#E50000', '#E50000']
    
    
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
    plt.figure(figsize = (10,6))
    color = ['#cccccc', '#cccccc', '#cccccc', '#cccccc', '#E50000', '#E50000', '#cccccc',
                     '#E50000', '#cccccc', '#cccccc']
    
     # set the font scale
    sns.set(font_scale= 1)
    
    sns.set_style('ticks')
    
    # if statement for which graph will be returned 
    if bad:
        bad_resid = data[data.bad_resident == 1]
        bar= sns.countplot(x= column, data=bad_resid, palette= color)
        plt.title(f'Number of Bad Residents by {column.capitalize()}')
        plt.xlabel(f'{column.capitalize()} Length')
        plt.ylabel('Total Count')
        bar = plt.gca()
        bar.set_facecolor('white')
        
        for p in bar.patches:
            height= p.get_height() / len(bad_resid) * 100
            bar.annotate(f"{round(height)}%", (p.get_x() + p.get_width() / 2,
                                            p.get_height()), ha='center', va='center',
                                            xytext=(0, 5), textcoords='offset points')
        plt.show()
    
    else:
        bar = sns.countplot(x=column, data=data, palette= color, ec= 'black')
        plt.title(f'Number of Bad Residents by {column.capitalize()}')
        plt.xlabel(f'{column.capitalize()} Length')
        plt.ylabel('Total Count')
        bar = plt.gca()
        bar.set_facecolor('white')

        for p in bar.patches:
            height = p.get_height() / len(data) * 100
            bar.annotate(f"{round(height)}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha= 'center', va= 'center', xytext= (0, 5), textcoords= 'offset points')
        plt.show()
    
    
def hist_data_inc(data, x_var, xlim=None, color = None):
    
    sns.set_style("white")
    plt.figure(figsize=(10,6))
    sns.histplot(data=data, x=x_var, color = color, edgecolor = 'black', linewidth = 1)
    plt.xlim(xlim)
    plt.title(f"{x_var.capitalize().split('_')[0]} {x_var.split('_')[1].capitalize()} Per Resident",
              fontsize= 13)
    plt.xlabel(f"{x_var.capitalize().split('_')[0]} {x_var.split('_')[1].capitalize()}", fontsize=12)
    plt.ylabel('Count', fontsize=12)
    return plt.show()
    
    
def countplot_a(data, column, color, bad = False):
    
    plt.figure(figsize = (10,6))
    bins = [300,400,500,600,700,800,900]
    if bad: 
        risk_bin = pd.cut(data[column], bins = bins)
        sns.countplot(x = risk_bin, hue = data.bad_resident,color = color, ec = 'black')
        plt.title(f'''Number Of Residents By {column.capitalize().split('_')[0]} {column.split('_')[1].capitalize()}''',
              fontsize= 13)
        plt.xlabel(f"{column.capitalize().split('_')[0]} {column.split('_')[1].capitalize()}",
                   fontsize= 12)
        plt.ylabel('Count', fontsize= 12)
        plt.legend(labels= ['Good Resident','Bad Resident'], loc = 'upper left')
        plt.show()
    
    else:
        bad_risk = pd.cut(data[data.bad_resident == 1].risk_score, bins = bins)
        sns.countplot(x = bad_risk, color = color, ec = 'black', alpha = 1)
        plt.title(f'''Number Of Residents By {column.capitalize().split('_')[0]} {column.split('_')[1].capitalize()}''',
                  fontsize= 13)
        plt.xlabel(f"{column.capitalize().split('_')[0]} {column.split('_')[1].capitalize()}",
                   fontsize= 12)
        plt.ylabel('Count', fontsize = 12)
        plt.show()  
        
def viz_bad_resident_risk(train):
    
    '''Plot precentage of bad resident within risk score'''
    
    r1 = len(train[(train.risk_score < 301) & (train.bad_resident == 1)]) / len(train.risk_score[train.risk_score < 301])
    r2 = len(train[(train.risk_score >300) & (train.risk_score < 401) & (train.bad_resident == 1)]) / len(train[(train.risk_score >300) & (train.risk_score < 401)])
    r3 = (len(train[(train.risk_score >400) & (train.risk_score < 501) & (train.bad_resident == 1)]) /
    len(train[(train.risk_score >400) & (train.risk_score < 501)]))
    r4 = (len(train[(train.risk_score >500) & (train.risk_score < 601) & (train.bad_resident == 1)]) /
    len(train[(train.risk_score >500) & (train.risk_score < 601)]))
    r5 = (len(train[(train.risk_score >600) & (train.risk_score < 701) & (train.bad_resident == 1)]) /
    len(train[(train.risk_score >600) & (train.risk_score < 701)]))
    r6= (len(train[(train.risk_score >700) & (train.risk_score < 801) & (train.bad_resident == 1)]) /
    len(train[(train.risk_score >700) & (train.risk_score < 801)]))
    r7 = (len(train[(train.risk_score >800) & (train.risk_score < 100000000) & (train.bad_resident == 1)]) /
    len(train[(train.risk_score >800) & (train.risk_score < 100000000)]))
    rs = pd.DataFrame(data = {'0-300': r1, '400-500': r3, '500-600': r4,
                         '600-700': r5, '700-800': r6, '>800':r7}, index = [0])
    rs = rs.round(2) * 100
    
    plt.figure(figsize= (10,6))
    sns.set_style("white")
    color= ['#CCCCCC', 'red', '#CCCCCC', '#CCCCCC', '#CCCCCC']
    bar= sns.barplot(data= rs, palette= color, edgecolor=['#CCCCCC', 'black', '#CCCCCC','#CCCCCC', '#CCCCCC'])
    plt.xlabel('Risk Score Range')
    plt.ylabel('Percent')
    plt.title('Relationship Between Risk Score And "Bad Resident"?')
    bar.set_xticklabels(bar.get_xticklabels(), rotation= 50)
    for p in bar.patches:
            bar.annotate(f"{round(p.get_height())}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize= 13)
    plt.show()

        
        
def chi_test_a(data, column, risk = False):
    '''get result of chi-square test'''
    
    
    if risk: 
        bins = [300,400,500,600,700,800,900]
        risk_bin = pd.cut(data[column], bins = bins)

        observed = pd.crosstab(data.bad_resident, risk_bin)
        chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    else:
        
        inc_bins = [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 10000000]
        month = pd.qcut(data[column], 4)
    
        observed = pd.crosstab(data.bad_resident, month)
        chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    𝜶 = .05

    if p < 𝜶:
        print("We reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")
    
    return print(f'''
Chi2 = {chi2:.3f}
P-value = {p:.3f}''')   

def viz_bad_resident_term(train):
    '''Plot precentage of bad resident within term'''

    [2,4,6,11,12,13,14,15,16,17,18]
    t1 = len(train[(train.term == 2) & (train.bad_resident == 1)])/ len(train[train.term == 2])
    t2 = len(train[(train.term == 4) & (train.bad_resident == 1)])/ len(train[train.term == 4])
    t3 = len(train[(train.term == 6) & (train.bad_resident == 1)])/ len(train[train.term == 6])
    t4 = len(train[(train.term == 11) & (train.bad_resident == 1)])/ len(train[train.term == 11])
    t5 = len(train[(train.term == 12) & (train.bad_resident == 1)])/ len(train[train.term == 12])
    t6 = len(train[(train.term == 13) & (train.bad_resident == 1)])/ len(train[train.term == 13])
    t7 = len(train[(train.term == 14) & (train.bad_resident == 1)])/ len(train[train.term == 14])
    t8 = len(train[(train.term == 15) & (train.bad_resident == 1)])/ len(train[train.term == 15])
    t9 = len(train[(train.term == 16) & (train.bad_resident == 1)])/ len(train[train.term == 16])
    t10 = len(train[(train.term == 17) & (train.bad_resident == 1)])/ len(train[train.term == 17])
    t11 = len(train[(train.term == 18) & (train.bad_resident == 1)])/ len(train[train.term == 18])
    term = pd.DataFrame(data = {'1': t1, '4': t2, '6': t3, '11':t4, '12':t5, '13':t6, '14':t7,
                               '15':t8, '16':t9, '17':t10, '18':t11}, index = [0])
    term = term.round(2) * 100
     
    plt.figure(figsize = (10,6))
    color = ['#E50000', '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC',
                         '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC']
    # set the font scale
    sns.set(font_scale= 1)
    sns.set_style('ticks')
    bar= sns.barplot(data=term, palette= color, edgecolor= ['black', '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC',
                    '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC', '#CCCCCC'])
    plt.title(f'Number of Bad Residents by Term')
    plt.xlabel(f'Term in Months')
    plt.ylabel('Total Count')
    bar = plt.gca()
    bar.set_facecolor('white')
    
    for p in bar.patches:
            height= p.get_height() / len(term) * 1
            bar.annotate(f"{round(height)}%",(p.get_x() + p.get_width() / 2,
                                                 p.get_height()), ha='center', va='center',
                                                 xytext=(0, 5), textcoords='offset points')
    plt.show()


def plot_bad_propertiess(train):
    
    '''
    This function returns a visual of the states with the highest
    amount of charge codes.
    '''
    # Define a list of properties
    properties = ['Colorado', 'Texas', 'Georgia', 'Arizona', 'North Carolina']

    # Initialize an empty list to store the results
    results = []

    # Loop over each property and calculate the CO variable
    for prop in properties:
        CO = (round(len(train[(train.prop_id == prop) & (train.bad_resident == 1)])/ len(train[train.prop_id == prop]),
                   2) * 100)
        results.append(CO)
    
    # Sort the results in reverse order
    properties_sorted = [x for _, x in sorted(zip(results, properties), reverse=True)]
    results_sorted = sorted(results, reverse=True)

    # Create a pandas DataFrame with the results
    df = pd.DataFrame(data = {'Properties': properties_sorted, 'CO': results_sorted})
    
    # Setting the palette order 
    cl= ['#E50000', '#cccccc', '#cccccc', '#cccccc', '#cccccc']
    
    plt.figure(figsize = (10,6))
    
    # Creating the graph
    bar = sns.barplot(data=df, x='Properties', y='CO', palette=cl, order=properties_sorted)
    patch_h = [patch.get_height() for patch in bar.patches]   
    idx_tallest = np.argmax(patch_h)   
    
    # Setting the title
    plt.title('Properties With The Most Damage Codes')
    
    
    # Adding the total count number on the top of the bars
    for p in bar.patches:
        bar.annotate(f"{round(p.get_height())}%", (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    # Returning the graph
    plt.show()
    
