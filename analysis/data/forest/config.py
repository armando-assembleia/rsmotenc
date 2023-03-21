import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

###################
### Paths
###################
#ANALYSIS_ROOT = Path(os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir))
#ANALYSIS_ROOT = Path(Path(os.getcwd()) / 'analysis')
CUR_PATH = Path(__file__).parent.resolve()
ANALYSIS_ROOT = CUR_PATH.parents[1]
DATA = Path(ANALYSIS_ROOT / 'data' / 'forest')
MODELS = Path(ANALYSIS_ROOT / 'models' / 'forest')
REPORTS = Path(ANALYSIS_ROOT / 'reports')


####################
### Variable Details
####################

#Index of the continuous variables
idnum = [0,1,2,3,4,5,6,7,8,9]
#Index of the categorical variables
idcat = [10, 11]
#Index of the binary variables
idbin = []


####################
### Models Details
####################

## Split the data to be 5-fold cross-validated
kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

#randomforest model - hyperparameter tuning using grid search
param_grid = {'max_depth': [7,11],
              'max_features': [3,6],
              'min_samples_leaf': [3],
              'min_samples_split': [3],
              'n_estimators': [500]
}# Create a base model
param_grid = {'randomforestclassifier__' + key: param_grid[key] for key in param_grid}


techs = ["SMOTENC",
         "SMOTEENC",
         "RSMOTENC_gower",
         "RSMOTENC_huang",
         "RSMOTENC_ahmadA",
         "RSMOTENC_ahmadFD",
         "RSMOTENC_ahmadMahA",
         "RSMOTENC_ahmadMahFD",
         "RSMOTENC_ahmadL1A",
         "RSMOTENC_ahmadL1FD",]