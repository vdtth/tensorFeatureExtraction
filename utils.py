import os
import sys
import time
import glob

import pickle
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
import multiprocessing
import torch
import shutil
import tensorly as tl
from tensorly.metrics.regression import RMSE
from tensorly.decomposition import non_negative_tucker, non_negative_tucker_hals,tucker

def preprocess(image_path,rank,group,decomposition_type):
    #read image
    img = nib.load(image_path)
    img = img.get_fdata()[:,:,:]
    rank_str = '_'.join([str(elem) for elem in rank])
    
    #normalize image
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = torch.from_numpy(img)
    img = img.to('cuda')
    
    
    dirname = os.path.dirname(image_path)
    basename = os.path.basename(image_path).replace(".nii", f"_{decomposition_type}_{rank_str}.p")

    saveDir = os.path.join(dirname,f"../{group}_{decomposition_type}_{rank_str}")
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    saveFile = os.path.join(saveDir, basename)                                                
    if os.path.exists(saveFile): 
        return
    
    tl.set_backend('pytorch')
    tic = time.time()
    if decomposition_type.lower() == "ntf":
        tensor_mu, error_mu = non_negative_tucker(img, rank=rank, tol=1e-12, n_iter_max=1000, return_errors=True)
        tucker_reconstruction_mu = tl.tucker_to_tensor(tensor_mu)
        core = tensor_mu.core.cpu().numpy().tolist()
        factors =   tensor_mu.factors
        factors = [x.cpu().numpy() for x  in factors]

    if decomposition_type.lower() == "tucker":
        tensor_mu = tucker(img, rank=rank, tol=1e-12, n_iter_max=1000)
        core = tensor_mu.core.cpu().numpy().tolist()
        factors =   tensor_mu.factors
        factors = [x.cpu().numpy() for x  in factors]
    
   
    with open(saveFile,'wb') as f:
        pickle.dump([core, factors],f)

def decompose(img,rank):
    img = torch.from_numpy(img)
    img = img.to('cuda')
    decomposition_type = "tucker"

    tl.set_backend('pytorch')
    tic = time.time()
    if decomposition_type.lower() == "ntf":
        tensor_mu, error_mu = non_negative_tucker(img, rank=rank, tol=1e-12, n_iter_max=1000, return_errors=True)
        tucker_reconstruction_mu = tl.tucker_to_tensor(tensor_mu)
        core = tensor_mu.core.cpu().numpy().tolist()
        factors =   tensor_mu.factors
        factors = [x.cpu().numpy() for x  in factors]

    if decomposition_type.lower() == "tucker":
        tensor_mu = tucker(img, rank=rank, tol=1e-12, n_iter_max=1000)
        core = tensor_mu.core.cpu().numpy().tolist()
        factors =   tensor_mu.factors
        factors = [x.cpu().numpy() for x  in factors]
    time_mu = time.time()-tic
    print(f"Processing time for {decomposition_type} decomposition: {time_mu}")
    return core,factors



from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import ComplementNB
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree._classes import ExtraTreeClassifier
from sklearn.ensemble._forest import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process._gpc import GaussianProcessClassifier
from sklearn.ensemble._gb import GradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.neighbors._classification import KNeighborsClassifier
from sklearn.semi_supervised._label_propagation import LabelPropagation
from sklearn.semi_supervised._label_propagation import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm._classes import LinearSVC
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.linear_model._logistic import LogisticRegressionCV
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors._nearest_centroid import NearestCentroid
from sklearn.svm._classes import NuSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.linear_model._passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model._perceptron import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors._classification import RadiusNeighborsClassifier
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.linear_model._ridge import RidgeClassifier
from sklearn.linear_model._ridge import RidgeClassifierCV
from sklearn.linear_model._stochastic_gradient import SGDClassifier
from sklearn.svm._classes import SVC
from sklearn.ensemble._stacking import StackingClassifier
from sklearn.ensemble._voting import VotingClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score,fbeta_score
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def run_exps(X_train: pd.DataFrame , y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    '''
    Lightweight script to test many models and find winners
:param X_train: training split
    :param y_train: training target vector
    :param X_test: test split
    :param y_test: test target vector
    :return: DataFrame of predictions
    '''
    
    dfs = []
    models = [
                ("KNeighborsClassifier",KNeighborsClassifier()),
                ("LogisticRegression",LogisticRegression()),
                ("LinearSVC",LinearSVC()),
                ("DecisionTreeClassifier",DecisionTreeClassifier()),
                ("RandomForestClassifier",RandomForestClassifier()),
#                 ("AdaBoostClassifier",AdaBoostClassifier()),
#                 ("BaggingClassifier",BaggingClassifier()),
#                 ("BernoulliNB",BernoulliNB()),
#                 ("CalibratedClassifierCV",CalibratedClassifierCV()),
#                 ("CategoricalNB",CategoricalNB()),
#                 ("ClassifierChain",ClassifierChain(LogisticRegression())),
#                 ("ComplementNB",ComplementNB()),
#                 ("DummyClassifier",DummyClassifier()),
#                 ("ExtraTreeClassifier",ExtraTreeClassifier()),
#                 ("ExtraTreesClassifier",ExtraTreesClassifier()),
#                 ("GaussianNB",GaussianNB()),
#                 ("GaussianProcessClassifier",GaussianProcessClassifier()),
#                 ("GradientBoostingClassifier",GradientBoostingClassifier()),
#                 ("HistGradientBoostingClassifier",HistGradientBoostingClassifier()),
#                 ("LabelPropagation",LabelPropagation()),
#                 ("LabelSpreading",LabelSpreading()),
#                 ("LinearDiscriminantAnalysis",LinearDiscriminantAnalysis()),
#                 ("LogisticRegressionCV",LogisticRegressionCV()),
#                 ("MLPClassifier",MLPClassifier()),
#                 ("MultiOutputClassifier",MultiOutputClassifier(KNeighborsClassifier())),
#                 ("MultinomialNB",MultinomialNB()),
#                 ("NearestCentroid",NearestCentroid()),
#                 ("NuSVC",NuSVC()),
#                 ("OneVsOneClassifier",OneVsOneClassifier(LinearSVC())),
#                 ("OneVsRestClassifier",OneVsRestClassifier(SVC())),
#                 ("OutputCodeClassifier",OutputCodeClassifier(estimator=RandomForestClassifier())),
#                 ("PassiveAggressiveClassifier",PassiveAggressiveClassifier()),
#                 ("Perceptron",Perceptron()),
#                 ("QuadraticDiscriminantAnalysis",QuadraticDiscriminantAnalysis()),
#                 ("RadiusNeighborsClassifier",RadiusNeighborsClassifier()),
#                 ("RidgeClassifier",RidgeClassifier()),
#                 ("RidgeClassifierCV",RidgeClassifierCV()),
#                 ("SGDClassifier",SGDClassifier()),
#                 ("SVC",SVC())
                # ("StackingClassifier",StackingClassifier())   
            ]
    results = []
    names = []
    f1 = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    
    """
    !!!!!!!!!! Change target names
    """
    target_names = ['AD', 'HC']
    for name, model in tqdm(models):
#             try:
                print(name)
                clf = model.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                # print(classification_report(y_test, y_pred, target_names=target_names))
                f1.append([name,accuracy_score(y_test, y_pred), precision_score(y_test, y_pred),
                           recall_score(y_test, y_pred),fbeta_score(y_test, y_pred,beta=1)])
                names.append(name)
                this_df = pd.DataFrame()
                this_df['model'] = name
                dfs.append(this_df)
                # print(f1)
                # print(dfs)
#             except: 
#                 continue
    f = pd.DataFrame(f1,columns=['Model','Accuracy','Precision','Recall','F-score'])
    final = pd.concat(dfs, ignore_index=True)
    return final,f