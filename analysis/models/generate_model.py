DATASET_NAME = "rain"

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import make_pipeline
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

import sys
#sys.path.append("../..")

from utils.RSMOTENC import RSMOTENC
from utils.SMOTEENC import SMOTEENC
from utils.auxSamplingStudy import *

class BalanceModel:
    
    def __init__(self,
                 dataset_name,
                 X_train,
                 y_train,
                 idcat) -> None:
        
        self.dataset_name = dataset_name
        self.X = X_train
        self.y = y_train
        self.idcat = idcat
        
        self.nbinsA = self.calculate_nbins(method = "average")
        self.nbinsFD = self.calculate_nbins(method = "FD")
        
        self.dict_RSMOTENC_methods = {"RSMOTENC_gower": {"method": "gower", "weigths_boolean": False, "nbins":3},
                            "RSMOTENC_huang": {"method": "huang", "weigths_boolean": False, "nbins":3},
                            "RSMOTENC_ahmadA": {"method": "ahmad", "weigths_boolean": True, "nbins":self.nbinsA},
                            "RSMOTENC_ahmadFD": {"method": "ahmad", "weigths_boolean": True, "nbins":self.nbinsFD},
                            "RSMOTENC_ahmadMahA": {"method": "ahmad_mahalanobis", "weigths_boolean": True, "nbins":self.nbinsA},
                            "RSMOTENC_ahmadMahFD": {"method": "ahmad_mahalanobis", "weigths_boolean": True, "nbins":self.nbinsFD},
                            "RSMOTENC_ahmadL1A": {"method": "ahmad_l1", "weigths_boolean": True, "nbins":self.nbinsA},
                            "RSMOTENC_ahmadL1FD": {"method": "ahmad_l1", "weigths_boolean": True, "nbins":self.nbinsFD}
                            } 
        
    def calculate_nbins(self, method = "FD"):
        n = self.X.shape[0]
        cat_vars = self.X.columns[self.idcat]
        
        if method == "root":
            return int(n**(0.5))
        
        elif method == "average":
            nbins_list = []
            for cat in cat_vars:
                nbins_list.append(len(self.X[cat].unique()))
            #print(nbins_list)
            return int(np.mean(nbins_list))
        
        elif method == "FD":
            nbins_list = []
            for cat in cat_vars:
                iqr = np.quantile(self.X[cat], 0.75) - np.quantile(self.X[cat], 0.25)
                h = 2 * iqr * n**(-1/3)
                nbins_list.append( ( np.max(self.X[cat]) - np.min(self.X[cat]) ) / h )
            #print(nbins_list)
            return int(np.median(nbins_list))
        
    def _balance_tech(self, tech_name):
        if tech_name == "SMOTENC":
            tech = SMOTENC(self.idcat)
        elif tech_name == "SMOTEENC":
            tech = SMOTEENC(self.idcat)
        elif tech_name[:8] == "RSMOTENC":
            dict_RSMOTENC = self.dict_RSMOTENC_methods[tech_name]
            tech = RSMOTENC(cat_vars = X_train.columns[self.idcat],
                            method = dict_RSMOTENC["method"],
                            weigths_boolean = dict_RSMOTENC["weigths_boolean"],
                            nbins = dict_RSMOTENC["nbins"])
        else:
            print(f'{tech_name} is not a valid balancing techinique name')
        return tech
    
    
    def _generate_model(self, tech, param_grid, kfold, path) -> None:
        
        filename = (path / f'{self.dataset_name}_{tech}.sav')
        samp_pipeline = make_pipeline(self._balance_tech(tech), 
                                    RandomForestClassifier(random_state=42))
        # check model performance on different values of hyper-parameters.
        grid_search = GridSearchCV(samp_pipeline, param_grid=param_grid, cv=kfold, scoring='balanced_accuracy',
                                return_train_score=True, n_jobs = 1, verbose = 0)
        grid_search.fit(self.X, self.y)
        best_grid = grid_search.best_estimator_
        pickle.dump(best_grid, open(filename, 'wb'))
        
        #print(f'{tech}: Sucessfully saved in {filename}' )
    
    def generate_models(self, techs, param_grid, kfold, path) -> None:
        
        for tech in techs:
            try:
                self._generate_model(tech, param_grid, kfold, path)
                #print(tech, u'\u2713', 'Sucessfully saved')
                print(tech, '✅', '\033[92m' + 'Sucessfully saved'  + '\033[0m')
            except ValueError:
                print(tech, '❌', '\033[91m' + 'FAILED'  + '\033[0m')      

sys.path.append(f'analysis/data/{DATASET_NAME}')
from config import DATA, MODELS, idcat, idnum, param_grid, kfold, techs
sys.path.remove(f'analysis/data/{DATASET_NAME}')


X_train = pd.read_csv(DATA / f"{DATASET_NAME}_X_train.csv")
y_train = pd.read_csv(DATA / f"{DATASET_NAME}_y_train.csv")

BalanceModel(DATASET_NAME,
             X_train,
             y_train,
             idcat
             ).generate_models(techs,
                               param_grid,
                               kfold,
                               MODELS)
             
#example

            