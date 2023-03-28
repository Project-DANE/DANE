import pandas as pd
import numpy as np

import wrangle as w
import explore as e
import modeling as m

import seaborn as sns
import matplotlib.pyplot as plt 


from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier


from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.utils import resample
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN


# Logistic regression Model
def logistic_regression_model(train_scaled, y_train, val_scaled, y_val): 
    
    # logistic regression model with the train dataset 
    seed = 91
    logit = LogisticRegression(random_state = seed)
    logit.fit(train_scaled, y_train)

    y_pred = logit.predict(train_scaled)
    train_accuracy = logit.score(train_scaled, y_train)

    # logistic regression model with the validate dataset

    logit = LogisticRegression(random_state = seed)
    logit.fit(val_scaled, y_val)

    y_pred_val = logit.predict(val_scaled)
    val_accuracy = logit.score(val_scaled, y_val)
    
    class_report_train = classification_report(y_train, y_pred)
    class_report_val = classification_report(y_val, y_pred_val)
    
    # Extract the recall for each class
    recall_train = pd.Series(class_report_train.split()[6:13:5], name='train_log')
    recall_validate = pd.Series(class_report_val.split()[6:13:5], name='validate_log')
    df = pd.concat([recall_train, recall_validate], axis=1)
    df = pd.DataFrame(df, columns= ['train_log', 'validate_log'])

    return df


def upsample(train):
    '''takes a train dataframe, upsamples minority and returns upsampled X_train and upsampled y_train'''
    
    
    df_majority = train[train.bad_resident ==0]
    df_minority = train[train.bad_resident ==1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=300 , random_state=91)
    
    # get upsampled_df
    df_upsampled = pd.concat([df_minority_upsampled, df_majority])
    
    # scale df_upsampled
    df_upsampled[['rent', 'monthly_income', 'total_income','age', 'risk_score']] = StandardScaler().fit_transform(df_upsampled[['rent', 'monthly_income', 'total_income','age', 'risk_score']])
    
    # split scaled df_upsampled
    X_train_upsampled = df_upsampled.drop('bad_resident',axis=1)
    y_train_upsampled = df_upsampled['bad_resident']
    
    return X_train_upsampled, y_train_upsampled


def get_knn(train, X_validate, y_validate):
    ''' takes a train dataframe, X_ validate, y_validate, print KNN  confusion matrix and classifaction report on train and validate data'''
    
    # get upsampled X_train and Y_train
    X_train_upsampled, y_train_upsampled = upsample(train)
    
    # create model
    knn= KNeighborsClassifier(n_neighbors =3)

    # fit the model to train data
    knn.fit(X_train_upsampled, y_train_upsampled)
    
    # make prediction on train obeservations
    y_pred = knn.predict(X_train_upsampled)
    
    # make prediction on validate obeservations
    y_pred_val = knn.predict(X_validate)
    
    # get confusion matrix
    confusion_matrix_train = metrics.confusion_matrix(y_train_upsampled, y_pred)
    confusion_matrix_val = metrics.confusion_matrix(y_validate, y_pred_val) 
    
    df1 = pd.DataFrame(classification_report(y_train_upsampled, y_pred, labels=['0','1'], output_dict=True)).T
    df2 = pd.DataFrame(classification_report(y_validate, y_pred_val, labels=['0','1'], output_dict=True)).T
    
    train_df = df1.iloc[0:2,1:2]
    validate_df = df2.iloc[0:2,1:2]
    
    train_df.rename(columns = {'recall': 'recall_train_KNN'}, inplace=True)
    validate_df.rename(columns = {'recall': 'recall_validate_KNN'}, inplace=True)

    concated_df = pd.concat([train_df,validate_df],axis=1) 
    
    return concated_df

def randomforestNC(train, y_train, x, y):
    sm = SMOTENC(categorical_features= list(range(6,24)), 
             random_state = 91, sampling_strategy = .1)
    train_up, y_up = sm.fit_resample(train, y_train)
    
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight= 'balanced', 
                            criterion='entropy',
                            min_samples_leaf=3,
                            n_estimators=200,
                            max_depth=4, 
                            random_state=91)
    rf = rf.fit(train_up, y_up)
    train_pred = rf.predict(train_up)
    y_pred = rf.predict(x)
    class_report_train = classification_report(y_up, train_pred)
    class_report_val = classification_report(y, y_pred)
    # Extract the recall for each class
    recall_train = pd.Series(class_report_train.split()[6:13:5], name='train_rfnc')
    recall_validate = pd.Series(class_report_val.split()[6:13:5], name='validate_rfnc')
    df = pd.concat([recall_train, recall_validate], axis=1)
    df = pd.DataFrame(df, columns= ['train_rfnc', 'validate_rfnc'])
    
    return df
    
    
def randomforestSMOTE(train, y_train, x, y):
    sm = SMOTE(random_state = 91, sampling_strategy = .1)
    train_up, y_up = sm.fit_resample(train, y_train)
    
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight= 'balanced', 
                            criterion='entropy',
                            min_samples_leaf=3,
                            n_estimators=200,
                            max_depth=4, 
                            random_state=91)
    rf = rf.fit(train_up, y_up)
    train_pred = rf.predict(train_up)
    y_pred = rf.predict(x)
    class_report_train = classification_report(y_up, train_pred)
    class_report_val = classification_report(y, y_pred)
    # Extract the recall for each class
    recall_train = pd.Series(class_report_train.split()[6:13:5], name='train_rfsmote')
    recall_validate = pd.Series(class_report_val.split()[6:13:5], name='validate_rfsmote')
    df = pd.concat([recall_train, recall_validate], axis=1)
    df = pd.DataFrame(df, columns= ['train_rfsmote', 'validate_rfsmote'])

    return df    


def randomforestADASYN(train, y_train, x, y):
    ad = ADASYN(random_state = 91, sampling_strategy = .1)
    train_up, y_up = ad.fit_resample(train, y_train)
    
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight= 'balanced', 
                            criterion='entropy',
                            min_samples_leaf=3,
                            n_estimators=200,
                            max_depth=4, 
                            random_state=91)
    rf = rf.fit(train_up, y_up)
    train_pred = rf.predict(train_up)
    y_pred = rf.predict(x)
    class_report_train = classification_report(y_up, train_pred)
    class_report_val = classification_report(y, y_pred)
    # Extract the recall for each class
    recall_train = pd.Series(class_report_train.split()[6:13:5], name='train_rfasy')
    recall_validate = pd.Series(class_report_val.split()[6:13:5], name='validate_rfasy')
    df = pd.concat([recall_train, recall_validate], axis=1)
    df = pd.DataFrame(df, columns= ['train_rfasy', 'validate_rfasy'])
    
    return df
  
    
def xgboostSMOTE(train, y_train, x, y):
    sm = SMOTE(random_state = 91, sampling_strategy = .1)
    train_up, y_up = sm.fit_resample(train, y_train)
    cv = xgb.XGBClassifier(scale_pos_weight = 20, n_estimators = 200, max_depth = 4, max_leaves = 3, learning_rate = .2, 
                       min_child_weight = 1, 
                       eval_metric = roc_auc_score, early_stopping_rounds = 1)
    cv = cv.fit(X = train_up, y = y_up, eval_set=[(train_up, y_up), (x, y)])
    y_pred = cv.predict(x)
    train_pred = cv.predict(train_up)
    class_report_train = classification_report(y_up, train_pred)
    class_report_val = classification_report(y, y_pred)
    # Extract the recall for each class
    recall_train = pd.Series(class_report_train.split()[6:13:5], name='train_xgsmote')
    recall_validate = pd.Series(class_report_val.split()[6:13:5], name='validate_xgsmote')
    df = pd.concat([recall_train, recall_validate], axis=1)
    df = pd.DataFrame(df, columns= ['train_xgsmote', 'validate_xgsmote'])
    
    return df
    
    
def xgboostSMOTENC(train, y_train, x, y):
    sm = SMOTENC(categorical_features= list(range(6,24)), 
             random_state = 91, sampling_strategy = .1)    
    train_up, y_up = sm.fit_resample(train, y_train)
    cv = xgb.XGBClassifier(scale_pos_weight = 20, n_estimators = 200, max_depth = 4, max_leaves = 3, learning_rate = .2, 
                       min_child_weight = 1, 
                       eval_metric = roc_auc_score, early_stopping_rounds = 1)
    cv = cv.fit(X = train_up, y = y_up, eval_set=[(train_up, y_up), (x, y)])
    y_pred = cv.predict(x)
    train_pred = cv.predict(train_up)
    class_report_train = classification_report(y_up, train_pred)
    class_report_val = classification_report(y, y_pred)
    # Extract the recall for each class
    recall_train = pd.Series(class_report_train.split()[6:13:5], name='train_xgnc')
    recall_validate = pd.Series(class_report_val.split()[6:13:5], name='validate_xgnc')
    df = pd.concat([recall_train, recall_validate], axis=1)
    df = pd.DataFrame(df, columns= ['train_xgnc', 'validate_xgnc'])
    
    return df
    
    
def xgboostADASYN(train, y_train, x, y):
    ad = ADASYN(random_state = 91, sampling_strategy = .1)
    train_up, y_up = ad.fit_resample(train, y_train)
    cv = xgb.XGBClassifier(scale_pos_weight = 20, n_estimators = 200, max_depth = 4, max_leaves = 3, learning_rate = .2, 
                            min_child_weight = 1, 
                            eval_metric = roc_auc_score, early_stopping_rounds = 1)
    cv = cv.fit(X = train_up, y = y_up, eval_set=[(train_up, y_up), (x, y)])
    y_pred = cv.predict(x)
    train_pred = cv.predict(train_up)
    class_report_train = classification_report(y_up, train_pred)
    class_report_val = classification_report(y, y_pred)
    # Extract the recall for each class
    recall_train = pd.Series(class_report_train.split()[6:13:5], name='train_xgasy')
    recall_validate = pd.Series(class_report_val.split()[6:13:5], name='validate_xgasy')
    df = pd.concat([recall_train, recall_validate], axis=1)
    df = pd.DataFrame(df, columns= ['train_xgasy', 'validate_xgasy'])

    return df


def ensemble(train, y_train, x, y):
    ez = EasyEnsembleClassifier(n_estimators = 10, sampling_strategy = .5, random_state = 91)
    ez = ez.fit(train, y_train)
    y_pred = ez.predict(x)
    train_pred = ez.predict(train)
    class_report_train = classification_report(y_train, train_pred)
    class_report_val = classification_report(y, y_pred)
    # Extract the recall for each class
    recall_train = pd.Series(class_report_train.split()[6:13:5], name='train_ensemble')
    recall_validate = pd.Series(class_report_val.split()[6:13:5], name='validate_ensemble')
    df = pd.concat([recall_train, recall_validate], axis=1)
    df = pd.DataFrame(df, columns= ['train_ensemble', 'validate_ensemble'])

    return df

def comparing_models(train, train_scaled, y_train, val_scaled, y_val):
    
    log_model = logistic_regression_model(train_scaled, y_train, val_scaled, y_val).iloc[1]
    knn_model= get_knn(train, val_scaled, y_val).iloc[1]
    rf_nc_model = randomforestNC(train_scaled, y_train, val_scaled, y_val).iloc[1]
    rf_smote_model = randomforestSMOTE(train_scaled, y_train, val_scaled, y_val).iloc[1]
    rf_adasyn_model = randomforestADASYN(train_scaled, y_train, val_scaled, y_val).iloc[1]
    xg_smote_model = xgboostSMOTE(train_scaled, y_train, val_scaled, y_val).iloc[1]
    xg_smote_nc_model = xgboostSMOTENC(train_scaled, y_train, val_scaled, y_val).iloc[1]
    xg_adasyn_model = xgboostADASYN(train_scaled, y_train, val_scaled, y_val).iloc[1]
    ensemble_model = ensemble(train_scaled, y_train, val_scaled, y_val).iloc[1]
    
    
    
    models = [log_model, knn_model, rf_nc_model, rf_smote_model, rf_adasyn_model,
              xg_smote_model, xg_smote_nc_model, xg_adasyn_model, ensemble_model]

    # Create an empty DataFrame with the desired columns
    results_df = pd.DataFrame(columns=['Train Recall', 'Validation Recall'])

    for model in models:
        # Get the transpose of the model DataFrame
        m = pd.DataFrame(model)
        transpose = m.T

        # Extract the metrics from the values
        train_recall, val_recall = transpose.values[0]

        # Append the metrics to the results DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([[train_recall, val_recall]],
                                                          columns=['Train Recall', 'Validation Recall'])])

    # Reset the index of the results DataFrame
    results_df = results_df.reset_index(drop=True)
    results_df.index = ['log_model', 'knn_model', 'rf_nc_model', 'rf_smote_model', 'rf_adasyn_model',
              'xg_smote_model', 'xg_smote_nc_model', 'xg_adasyn_model', 'ensemble_model']
    results_df = results_df.astype('float')
    results_df = results_df.reset_index()
    results_df.rename(columns= {'index': 'Model'}, inplace = True)
    results_df= results_df.sort_values('Validation Recall', ascending= False)
    results_df = results_df.reset_index(drop = 'index')
    
    return results_df

def models(df):
    plt.figure(figsize=(12, 8))
    sns.set_style('white')
    cl = ['#cccccc', 'green']

    ax = sns.barplot(x='Model', y='value', hue='variable', 
                     data=pd.melt(df, id_vars=['Model'],
                                  var_name='variable', value_name='value'),
                    palette = cl)
    ax.set_title('Train and Validation Recall by Model', fontsize=20)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Recall', fontsize=14)
    ax.legend(fontsize=14, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_ylim(0, ax.get_ylim()[1]*1.1)
    plt.annotate('Best Model', xy=(2, .7),xytext=(1.75,.85),color='black', fontsize = 20, arrowprops = dict(facecolor ='black', shrink = 0.05))
    for p in ax.containers:
        ax.bar_label(p, label_type='edge', labels=[f"{int(height*100)}%" for height in p.datavalues], fontsize = 20)

    plt.legend(loc = 'best')
    
    return plt.show()


def best_model(train, y_train, x, y, x1, y1):
    ez = EasyEnsembleClassifier(n_estimators = 10, sampling_strategy = .5, random_state = 91)
    ez = ez.fit(train, y_train)
    y_pred = ez.predict(x)
    
    train_pred = ez.predict(train)
    y_preds = ez.predict(x1)
    
    class_report_train = classification_report(y_train, train_pred)
    class_report_val = classification_report(y, y_pred)
    class_report_test = classification_report(y1, y_preds)
    
    # Extract the recall for each class
    recall_train = pd.Series(class_report_train.split()[11], name='train_ensemble')
    recall_validate = pd.Series(class_report_val.split()[11], name='validate_ensemble')
    recall_test = pd.Series(class_report_test.split()[11], name = 'test_ensemble')
    
    df = pd.concat([recall_train, recall_validate, recall_test], axis=1)
    df = pd.DataFrame(df, columns= ['train_ensemble', 'validate_ensemble', 'test_ensemble'])

    return df


def best_model_bc():
    df = pd.DataFrame({'model': 'ensemble', 'train': [0.66], 'validate': [0.63], 'test': [0.55]})
    df.index = ['']

    # Reshape dataframe
    df = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['train', 'validate', 'test'])
    df.columns = ['index', 'variable', 'value']

    cl = ['#C9E4CA', '#6BBBA1', '#0C5449']

    sns.set_style("white")
    plt.figure(figsize=(10,6))
    plt.subplots_adjust(top = 1.3)
    ax = sns.barplot(x='index', y='value', hue='variable', data=df, linewidth=2,
                     ec = 'black', palette= cl, alpha = .7)
    plt.title('Recall By Data Subset', fontsize=18)
    plt.xlabel('Ensemble Model', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for p in ax.containers:
            ax.bar_label(p, label_type='edge',
                         labels=[f"{int(height*100)}%" for height in p.datavalues], fontsize = 15)

    plt.show()
    
    
def best_models(df):
    f= df.iloc[2:5]
    g = df.iloc[7:]

    df = pd.concat([f, g])

    df['Model'] = df['Model'].replace({'xg_smote_nc_model':'XG Boost',
                                     'rf_adasyn_model': 'Random Forest', 
                                     'ensemble_model': 'Ensemble', 
                                     'knn_model': 'KNN', 
                                     'log_model': 'Logistic Regression'})
    
    return df    