from re import L
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def remove_outlier(df: pd.DataFrame, column: str, outlier_assumption: float) -> pd.DataFrame:
    mean = np.mean(df[column])
    std = np.std(df[column])
    
    minimum = mean - outlier_assumption * std
    maximum = mean + outlier_assumption * std
    
    is_outlier = (df[column] < minimum) | (df[column] > maximum)

    df = df[~is_outlier]

    
    return df

# Label encode categorical variables
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self 

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

def evaluate_metrics(y_true, y_predict, output = False):

    from sklearn.metrics import confusion_matrix

    # Given the true class and the predicted class, this function yields the accuracy, precision, sensitivity and specificity
    ### y_true: boolean list of the true classes
    ### y_predict: boolean list of the predicted classes
    ### output: if output=True, it will print the confusion matrix and metrics associated to y_true and y_predict
    #
    ### OUTPUT: dictionary with four metrics - accuracy, precision, sensitivity and specificity

    results = {}

    tp, fn, fp, tn = confusion_matrix(y_true,y_predict,labels=[1,0]).reshape(-1)

    #cm = confusion_matrix(y_true,y_predict)
    cm = np.array([[tp,fp],[fn,tn]])
    results["Confusion Matrix"] = cm
    
    #tp = cm[0,0]
    #fp = cm[0,1]
    #fn = cm[1,0]
    #tn = cm[1,1]

    total1=sum(sum(cm))
    #####from confusion matrix calculate accuracy
    accuracy=(tp + tn)/total1
    results["Accuracy"] = accuracy

    if fn + tn == 0:
        precision_0 = 0
    else:
        precision_0 = tn/(fn + tn)
    results["Precision0"] = precision_0
    
    if fp + tn == 0:
        sensitivity_0 = 0
    else:
        sensitivity_0 = tn/(fp + tn)
    results["Sensitivity0"] = sensitivity_0

    if tp + fp == 0:
        precision_1 = 0
    else:
        precision_1 = tp/(tp + fp)
    results["Precision1"] = precision_1
    
    if tp + fn == 0:
        sensitivity_1 = 0
    else:
        sensitivity_1 = tp/(tp + fn)
    results["Sensitivity1"] = sensitivity_1

    if tn + fp == 0:
        specificity = 0
    else:
        specificity = tn/(tn + fp)
    results["Specificity"] = specificity
    
    if precision_0 + sensitivity_0 == 0:
        f1score_0 = 0
    else:
        f1score_0 = 2 * precision_0 * sensitivity_0 / (precision_0 + sensitivity_0)
    results["f1Score0"] = f1score_0

    if precision_1 + sensitivity_1 == 0:
        f1score_1 = 0
    else:
        f1score_1 = 2 * precision_1 * sensitivity_1 / (precision_1 + sensitivity_1)
    results["f1Score1"] = f1score_1

    IR = sum(y_true == 0) / sum(y_true == 1)
    
    f1score_IR = (IR * f1score_1 + f1score_0) / (IR + 1)
    results["f1ScoreIR"] = f1score_IR

    if (1/IR * sensitivity_1 + sensitivity_0) == 0:
        f1score_recallIR = 0
    else:
        f1score_recallIR = (1+1/IR) * sensitivity_1 * sensitivity_0 / ( 1/IR * sensitivity_1 + sensitivity_0)
    results["f1ScoreRecallIR"] = f1score_recallIR

    results = {key : np.around(results[key], 3) for key in results}

    if output:
        print('Confusion Matrix : \n', cm)
        print('Accuracy : ', results["Accuracy"])
        print('F1-Score IR : ', results["f1ScoreIR"])
        print('F1-Score Recall IR : ', results["f1ScoreRecallIR"])
        print('Class 1')
        print('Precision : ', results["Precision1"] )
        print('Sensitivity : ', results["Sensitivity1"] )
        print('F1-Score : ', results["f1Score1"])
        print('Class 0')
        print('Precision : ', results["Precision0"] )
        print('Sensitivity : ', results["Sensitivity0"] )
        print('F1-Score : ', results["f1Score0"])

    return results

def evaluate(x, y, loaded_model, threshold):
    x = np.array(x)
    y = np.ravel(y)
    pred = (loaded_model.predict_proba(x)[:,1] >= threshold).astype(bool)
    results = evaluate_metrics(y, pred, output = True)
    #print(pd.crosstab(y, pred, rownames=['Actual'], colnames=['Predicted']))
    #print(classification_report(y, pred,digits=4))
    return results

def evaluate_table(x, y, loaded_models, metrics, save=None):
    
    results_table = pd.DataFrame(np.zeros((len(loaded_models),len(metrics))))
    results_table.index = loaded_models.keys()
    results_table.columns = metrics
    
    for model_name, model in loaded_models.items():
        threshold = evaluate_best_threshold(x,y, model, param="f1ScoreIR")
        results = evaluate(x, y, model, threshold)
        print(results)
        results = [results[metric] for metric in metrics]
        results_table.loc[model_name] = results
    
    if save != None:
        results_table.to_excel(save + ".xlsx")
    
    return results_table

def evaluate_best_threshold(x,y, loaded_model, param="f1-score"):
    x = np.array(x)
    y = np.ravel(y)
    best_param = 0
    best_threshold = 0
    for threshold in np.arange(0,1,0.05):
        pred = (loaded_model.predict_proba(x)[:,1] >= threshold).astype(bool)
        opt_param = evaluate_metrics(y, pred, output = False)[param]
        try:
            opt_param = evaluate_metrics(y, pred, output = False)[param]
            #f1score = classification_report(y,pred, digits=4, output_dict = True)["1.0"][param]
        except:
            print("ERROR!!!")
            #f1score = classification_report(y,pred, digits=4, output_dict = True)["1"][param]
        if opt_param > best_param:
            best_param = opt_param
            best_threshold = threshold
    return best_threshold

## generate_curves function creates ROC-AUC and PR-AUC curve of the loaded model and compare that wth the random classifier
def generate_curves(dict_files, X_test, y_test, save=None):   

    name_old, name_new = dict_files.keys()
    filename_old, filename_new = dict_files.values()

    f = plt.figure(figsize=(10,4))
    ax1 = f.add_subplot(121)
    loaded_model_old = pickle.load(open(filename_old, 'rb'))
    loaded_model_new = pickle.load(open(filename_new, 'rb'))

    test_prob_old = loaded_model_old.predict_proba(X_test)[:, 1]
    test_prob_new = loaded_model_new.predict_proba(X_test)[:, 1]
    
    fpr_old, tpr_old, _ = roc_curve(y_test,  test_prob_old)
    roc_auc_old = roc_auc_score(y_test,  test_prob_old)
    ax1.plot([0, 1], [0, 1], linestyle='--',label ='random, auc = %.3f'% 0.5, c = 'blue')
    ax1.plot(fpr_old, tpr_old ,label =f'{name_old}, auc = %.3f'% roc_auc_old, c= 'green')
    
    fpr_new, tpr_new, _ = roc_curve(y_test,  test_prob_new)
    roc_auc_new = roc_auc_score(y_test,  test_prob_new)
    ax1.plot(fpr_new, tpr_new ,label =f'{name_new}, auc = %.3f'% roc_auc_new, c= 'red')
    
    ax1.legend(loc=4)

    ax1.set_title('ROC curve' ,fontsize=16)
    ax1.set_ylabel('True Positive Rate',fontsize=14)
    ax1.set_xlabel('False Positive Rate',fontsize=14)

    ax2 = f.add_subplot(122)
    

    precision_old, recall_old, _ = precision_recall_curve(y_test, test_prob_old)
    precision_new, recall_new, _ = precision_recall_curve(y_test, test_prob_new)
    
    auc_score_old = auc(recall_old, precision_old)
    auc_score_new = auc(recall_new, precision_new)
    
    random_auc = y_test.sum()/len(y_test)
    
    ax2.plot([0, 1], [random_auc, random_auc], linestyle='--', label ='random, auc = %.3f'% random_auc, c ='blue')
    ax2.plot(recall_old, precision_old, label = f'{name_old}, auc=%.3f'% auc_score_old, c = 'green')
    ax2.plot(recall_new, precision_new, label = f'{name_new}, auc=%.3f'% auc_score_new, c = 'red')
    
    ax2.set_title('Precision Recall curve' ,fontsize=16)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.set_xlabel('Recall',fontsize=14)
    ax2.legend(loc='best')
    plt.show()
    if save is not None:
        f.savefig(save, bbox_inches='tight')
    
    return None;

def calculate_nbins(data, cat_cols = None, method = None):
    n = data.shape[0]
    cat_vars = data.columns[cat_cols]
    if method == "root":
        return int(n**(0.5))
    elif method == "average":
        nbins_list = []
        for cat in cat_vars:
            nbins_list.append(len(data[cat].unique()))
        print(nbins_list)
        return int(np.mean(nbins_list))
    elif method == "FD":
        nbins_list = []
        for cat in cat_vars:
            iqr = np.quantile(data[cat], 0.75) - np.quantile(data[cat], 0.25)
            h = 2 * iqr * n**(-1/3)
            nbins_list.append( ( np.max(data[cat]) - np.min(data[cat]) ) / h )
        print(nbins_list)
        return int(np.median(nbins_list))
