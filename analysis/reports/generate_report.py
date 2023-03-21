from matplotlib import rcParams

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


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


class GenerateReport:
    
    def __init__(self, dataset_name, techs, path_models) -> None:
        self.dataset_name = dataset_name
        
        self.MODELS = path_models
        
        self.techs = techs
        self.loaded_models = self._load_models(techs)
        
    def _load_models(self, techs):
        loaded_models = {}
        
        for tech in techs:
            filename = (self.MODELS / f'{self.dataset_name}_{tech}.sav')
            loaded_models[tech] = pickle.load(open(filename, 'rb'))

        return loaded_models
    
    def plot_feature_importance(self, vars, tech, path_report) -> None:
        
        #filename = (self.MODELS / f'{self.dataset_name}_{tech}.sav')
        #loaded_model = pickle.load(open(filename, 'rb'))
        loaded_model = self.loaded_models[tech]
        var_imp = (pd.Series(loaded_model.steps[1][1].feature_importances_, index=vars).nlargest(20))
        var_imp_df = var_imp.reset_index()
        var_imp_df.columns = ['Variable', f'Importance using {tech}']
        var_imp_df.set_index('Variable', inplace=True)

        plt.figure(figsize=(10, 10))
        rcParams.update({'figure.autolayout': True})
        var_imp_df.plot(kind='barh').invert_yaxis()
        plt.savefig(path_report / f'{self.dataset_name}_{tech}.jpeg', bbox_inches='tight')
        
        return None

    def table_performance(self, x, y, metrics, path_report) -> None:
    
        results_table = pd.DataFrame(np.zeros((len(self.loaded_models),len(metrics))))
        results_table.index = self.loaded_models.keys()
        results_table.columns = metrics
        
        for model_name, model in self.loaded_models.items():
            threshold = evaluate_best_threshold(x,y, model, param="f1ScoreIR")
            results = evaluate(x, y, model, threshold)
            print(results)
            results = [results[metric] for metric in metrics]
            results_table.loc[model_name] = results
        
        results_table.to_excel(path_report + ".xlsx")
    
        return None

    def plot_comparision() -> None:
        N = 5
        menMeans = (20, 35, 30, 35, 27)
        womenMeans = (25, 32, 34, 20, 25)
        ind = np.arange(N) # the x locations for the groups
        width = 0.35
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(ind, menMeans, width, color='r')
        ax.bar(ind, womenMeans, width,bottom=menMeans, color='b')
        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
        ax.set_yticks(np.arange(0, 81, 10))
        ax.legend(labels=['Men', 'Women'])
        plt.show()
        

import numpy as np
import matplotlib.pyplot as plt
data = [[30, 25, 50, 20],
[40, 23, 51, 17],
[35, 22, 45, 19]]
X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
    
    