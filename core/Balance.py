import pandas as pd

from sklearn.metrics import confusion_matrix

import copy

import RSMOTENC

class BalanceDataset:
    def __init__(self, cat_vars, method = "ahmad", weigths_boolean = True, nbins=3):
        self.cat_vars = cat_vars
        self.method = method
        self.weigths_boolean = weigths_boolean
        self.nbins = nbins

    def fit_resample(self, X, y):

        cat_idx = []
        for i in self.cat_vars:
            cat_idx.append(X.columns.get_loc(i))

        data_aux = copy.deepcopy(X)
        data_aux.insert(0, "anomaly", y)
        data_aux = data_aux.to_numpy()
        data_Rsmote = RSMOTENC.RSmote(data_aux, cat_idx , ir=1, k=5, method = self.method, weigths_boolean = self.weigths_boolean, nbins = self.nbins).over_sampling()

        new_X = pd.DataFrame(data_Rsmote[:,1:], columns = X.columns)
        new_y = pd.DataFrame(data_Rsmote[:,0])

        new_X[self.cat_vars] = new_X[self.cat_vars].astype("category")

        #Change new_X and new_y column names, according to X and y, respectively
        
        return new_X, new_y