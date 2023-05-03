import os
import sys
import time
import glob
import argparse
import pickle
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from utils import *
import datetime
from tqdm import tqdm
from config import *

import torch
import tensorly as tl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--decomposeType", default="tucker",type=str)
	parser.add_argument("-p", "--path", default=fr'E:\Thanh\Projects\QG-Alzheirmer\tensorFeatureExtraction\data\\',type=str)  
	parser.add_argument('-r', '--rank', help='delimited list input', type=str,default="30,30,25")

	args = parser.parse_args()

	path = args.path
	decomposition_type = args.decomposeType 
	rank = [int(item) for item in args.rank.split(',')]
	rank_str = '_'.join([str(elem) for elem in rank])
	options = vars(args)

	pair = ["AD", "NC"]
	rank_str = '_'.join([str(elem) for elem in rank])
	# E:\Thanh\Projects\QG-Alzheirmer\tensorFeatureExtraction\data\AD_tucker_30_30_25\feature_AD_NC
	decomposition_type = "tucker"
	path_dict = {x:os.path.join(path,fr'{x}_{decomposition_type}_{rank_str}') for x in pair}
	core_tensor = {x:[] for x in pair}
	core = {x:[] for x in pair}
	factor0 = {x:[] for x in pair}
	factor1 = {x:[] for x in pair}
	factor2 = {x:[] for x in pair}



	for group in pair:
	    for path in tqdm(glob.glob(os.path.join(path_dict[group],'*.p'))):
	        # print(path)
	        with open(path, 'rb') as f:
	            data = pickle.load(f)
	            core_i = np.array(data[0]).reshape(-1,)
	            factor = data[1]
	            factor0_i = factor[0].reshape(-1,)
	            factor1_i = factor[1].reshape(-1,)
	            factor2_i = factor[2].reshape(-1,)
	            core[group].append(core_i)
	            factor0[group].append(factor0_i)
	            factor1[group].append(factor1_i)
	            factor2[group].append(factor2_i)
    # break

	X_factor0 = np.r_[np.array(factor0[pair[0]]), np.array(factor0[pair[1]])]
	X_factor1 = np.r_[np.array(factor1[pair[0]]), np.array(factor1[pair[1]])]
	X_factor2 = np.r_[np.array(factor2[pair[0]]), np.array(factor2[pair[1]])]
	X_core = np.r_[np.array(core[pair[0]]), np.array(core[pair[1]])]
	# X_core = np.r_[core_NC,core_AD]
	X = X_core
	X = np.c_[X_core,X_factor0,X_factor1,X_factor2]
	y = [0]*np.array(core[pair[0]]).shape[0] + [1]*np.array(core[pair[1]]).shape[0] 
	y = np.array(y)


	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	for i, (train_index, test_index) in tqdm(enumerate(kfold.split(X, y))):
	    X_train = X[train_index,:]
	    y_train = y[train_index]
	    
	    X_test = X[test_index,:]
	    y_test = y[test_index]
	    sc = MinMaxScaler()
	    X_train = sc.fit_transform(X_train)
	    X_test = sc.transform(X_test)
	    # Training through all classifier
	#     final,f = run_exps(X, y, X_test, y_test)
	    final,f = run_exps(X_train, y_train, X_test, y_test)
	    f["Fold"] = i
	    if i == 0:
	        f_combine = pd.DataFrame(f)
	    else:
	        f_combine = pd.concat([f_combine,f])
	f_combine = f_combine.sort_values(["Model","Fold"])

	f_combine.to_csv(f'{"_".join(pair)}_{decomposition_type}_{rank_str}.csv', index=False)