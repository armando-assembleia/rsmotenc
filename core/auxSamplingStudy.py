from re import L
import pandas as pd
import numpy as np
from core.Balance import BalanceDataset

def remove_outlier(df: pd.DataFrame, column: str, outlier_assumption: float) -> pd.DataFrame:
    mean = np.mean(df[column])
    std = np.std(df[column])
    
    minimum = mean - outlier_assumption * std
    maximum = mean + outlier_assumption * std
    
    is_outlier = (df[column] < minimum) | (df[column] > maximum)

    df = df[~is_outlier]

    
    return df

# Label encode categorical variables

from sklearn.preprocessing import LabelEncoder

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

from sklearn.metrics import classification_report


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
    evaluate_metrics(y, pred, output = True)
    #print(pd.crosstab(y, pred, rownames=['Actual'], colnames=['Predicted']))
    #print(classification_report(y, pred,digits=4))
    return None;


def evaluate_table(x, y, loaded_models, threshold, metrics, save=None):
    
    results_table = pd.DataFrame(np.zeros((len(loaded_models),len(metrics))))
    results_table.index = loaded_models.keys()
    results_table.columns = metrics
    
    for loaded_model in loaded_models:
        threshold = evaluate_best_threshold(x,y, loaded_model, param="f1-score")
        results = evaluate(x, y, loaded_model, threshold)
        results = [results[metric] for metric in metrics]
        results_table.loc[loaded_model] = results
    
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

import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, roc_auc_score


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


from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, _safe_indexing, sparsefuncs_fast, check_X_y, check_random_state
from scipy import sparse
from numbers import Integral
from collections import Counter

# Our New Proposed SMOTE Method
from scipy import stats
class MySMOTENC():
    
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        
    def chk_neighbors(self, nn_object, additional_neighbor):
        if isinstance(nn_object, Integral):
            return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
        #elif isinstance(nn_object, KNeighborsMixin):
        #    return clone(nn_object)
        #else:
        #    raise_isinstance_error(nn_name, [int, KNeighborsMixin], nn_object)     
    
    def generate_samples(self, X, nn_data, nn_num, rows, cols, steps, continuous_features_,):
        rng = check_random_state(42)

        diffs = nn_data[nn_num[rows, cols]] - X[rows]

        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            X_new = X[rows] + steps * diffs 

        X_new = (X_new.tolil() if sparse.issparse(X_new) else X_new)
        # convert to dense array since scipy.sparse doesn't handle 3D
        nn_data = (nn_data.toarray() if sparse.issparse(nn_data) else nn_data)

        all_neighbors = nn_data[nn_num[rows]]

        for idx in range(continuous_features_.size, X.shape[1]):

            mode = stats.mode(all_neighbors[:, :, idx], axis = 1)[0]

            X_new[:, idx] = np.ravel(mode)            
        return X_new
    
    def make_samples(self, X, y_dtype, y_type, nn_data, nn_num, n_samples, continuous_features_, step_size=1.0):
        random_state = check_random_state(42)
        samples_indices = random_state.randint(low=0, high=len(nn_num.flatten()), size=n_samples)    
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self.generate_samples(X, nn_data, nn_num, rows, cols, steps, continuous_features_)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        
        return X_new, y_new
    
    def cat_corr_pandas(self, X, target_df, target_column, target_value):
    # X has categorical columns
        categorical_columns = list(X.columns)
        X = pd.concat([X, target_df], axis=1)

        # filter X for target value
        is_target = X.loc[:, target_column] == target_value
        X_filtered = X.loc[is_target, :]

        X_filtered.drop(target_column, axis=1, inplace=True)

        # get columns in X
        nrows = len(X)
        encoded_dict_list = []
        nan_dict = dict({})
        c = 0
        imb_ratio = len(X_filtered)/len(X)
        OE_dict = {}
        
        for column in categorical_columns:
            for level in list(X.loc[:, column].unique()):
                
                # filter rows where level is present
                row_level_filter = X.loc[:, column] == level
                rows_in_level = len(X.loc[row_level_filter, :])
                
                # number of rows in level where target is 1
                O = len(X.loc[is_target & row_level_filter, :])
                E = rows_in_level * imb_ratio
                # Encoded value = chi, i.e. (observed - expected)/expected
                ENC = (O - E) / E
                OE_dict[level] = ENC
                
            encoded_dict_list.append(OE_dict)

            X.loc[:, column] = X[column].map(OE_dict)
            ### ERROR DETECTED HERE (to_numpy() missing)
            nan_idx_array = np.ravel(np.argwhere(np.isnan(X.loc[:, column].to_numpy())))
            
            if len(nan_idx_array) > 0 :
                nan_dict[c] = nan_idx_array
            c = c + 1
            X.loc[:, column].fillna(-1, inplace = True)
            
        X.drop(target_column, axis=1, inplace=True)
        return X, encoded_dict_list, nan_dict

    def fit_resample(self, X, y):
        X_cat_encoded, encoded_dict_list, nan_dict = self.cat_corr_pandas(X.iloc[:,np.asarray(self.categorical_features)], y, target_column='target', target_value=1)
        X_cat_encoded = np.array(X_cat_encoded)
        y = np.ravel(y)
        X = np.array(X)

        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {key: n_sample_majority - value for (key, value) in target_stats.items() if key != class_majority}

        n_features_ = X.shape[1]
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == 'bool':
            categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any([cat not in np.arange(n_features_) for cat in categorical_features]):
                raise ValueError('Some of the categorical indices are out of range. Indices'
                            ' should be between 0 and {}'.format(n_features_))
            categorical_features_ = categorical_features

        continuous_features_ = np.setdiff1d(np.arange(n_features_),categorical_features_)

        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_continuous = X[:, continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=['csr', 'csc'])
        X_minority = _safe_indexing(X_continuous, np.flatnonzero(y == class_minority))

        if sparse.issparse(X):
            if X.format == 'csr':
                _, var = sparsefuncs_fast.csr_mean_variance_axis0(X_minority)
            else:
                _, var = sparsefuncs_fast.csc_mean_variance_axis0(X_minority)
        else:
            var = X_minority.var(axis=0)
        median_std_ = np.median(np.sqrt(var))

        X_categorical = X[:, categorical_features_]
        X_copy = np.hstack((X_continuous, X_categorical))

        X_cat_encoded = X_cat_encoded * median_std_

        X_encoded = np.hstack((X_continuous, X_cat_encoded))
        X_resampled = X_encoded.copy()
        y_resampled = y.copy()


        for class_sample, n_samples in sampling_strategy.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X_encoded, target_class_indices)
            nn_k_ = self.chk_neighbors(5, 1)
            nn_k_.fit(X_class)

            nns = nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self.make_samples(X_class, y.dtype, class_sample, X_class, nns, n_samples, continuous_features_, 1.0)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
                sparse_func = 'tocsc' if X.format == 'csc' else 'tocsr'
                X_resampled = getattr(X_resampled, sparse_func)()
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))
            
        X_resampled_copy = X_resampled.copy()
        i = 0
        for col in range(continuous_features_.size, X.shape[1]):
            encoded_dict = encoded_dict_list[i]
            i = i + 1
            for key, value in encoded_dict.items():
                X_resampled_copy[:, col] = np.where(np.round(X_resampled_copy[:, col], 4) == np.round(value * median_std_, 4), key, X_resampled_copy[:, col])

        for key, value in nan_dict.items():
            for item in value:
                X_resampled_copy[item, continuous_features_.size + key] = X_copy[item, continuous_features_.size + key]

               
        X_resampled = X_resampled_copy   
        indices_reordered = np.argsort(np.hstack((continuous_features_, categorical_features_)))
        if sparse.issparse(X_resampled):
            col_indices = X_resampled.indices.copy()
            for idx, col_idx in enumerate(indices_reordered):
                mask = X_resampled.indices == col_idx
                col_indices[mask] = idx
            X_resampled.indices = col_indices
        else:
            X_resampled = X_resampled[:, indices_reordered]
        return X_resampled, y_resampled

from pyod.models.lof import LOF
import aux
 
def test_distances(distance_matrices, y, index, neigbs, outlier_prevelance, metric, save=None):
    for name_matrix, distance_matrix in distance_matrices.items():

        metric_list = []

        for i in neigbs:
            list_methods = [LOF(metric="precomputed", n_neighbors=i)]
            list_names = ["LOF"]

            data = aux.apply_methods(distance_matrix, list_methods, list_names, index, outlier_prevelance)
            data = aux.join_dfs([data])

            metric_list.append(aux.evaluate(y, data['LOF'])[metric])

        print(aux.evaluate(y, data['LOF'])[metric])
        
        plt.plot(neigbs, metric_list, label = name_matrix)

    plt.title('LOF(k)')
    plt.xlabel('k')
    plt.ylabel(metric)
    plt.legend()

    if save is not None:
        plt.savefig(save, bbox_inches='tight')

    plt.show()

    return None


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
