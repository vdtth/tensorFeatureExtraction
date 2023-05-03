import os
import sys
import time
import glob
import argparse
import pickle
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from utils import preprocess,decompose
import datetime
from tqdm import tqdm
from config import *

import torch
import tensorly as tl


import torch
import tensorly as tl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument("-d", "--decomposeType", default="tucker",type=str)
	# parser.add_argument("-p", "--path", default=fr'E:\Thanh\Projects\QG-Alzheirmer\tensorFeatureExtraction\data\\',type=str)  
	# parser.add_argument('-r', '--rank', help='delimited list input', type=str,default="30,30,25")

	# args = parser.parse_args()

	# path = args.path
	# decomposition_type = args.decomposeType 
	# rank = [int(item) for item in args.rank.split(',')]
	rank_str = '_'.join([str(elem) for elem in rank])
	# options = vars(args)



	# =============================
	print("Tensor compression")
	for group in ['AD', 'MCI', 'NC']:
	    print(f"group: {group}")
	    for image in tqdm(glob.glob(os.path.join(path,group,'*.nii'))):
	        preprocess(image,rank,group,decomposition_type)



	# ================
	combine_paths = ([glob.glob( path +fr'{x}_tucker_{rank_str}//*.p') for x in pair])
	combine_paths = combine_paths[0] + combine_paths[1]
	core_tensor = []
	factor_dict= {}
	for x in range(3):
	    factor_dict[f"factor{x}"] = []
	for path in tqdm(combine_paths):
	    with open(path, 'rb') as f:
	        data = pickle.load(f)
	        core =  np.array(data[0])
	        factors = data[1]
	        core_tensor.append(core)
	        [factor_dict[f"factor{x}"].append(factors[x]) for x in range(3)]

	Nsubject_core_tensor = np.array(core_tensor)
	Nsubject_factor0_tensor = np.array(factor_dict["factor0"])
	Nsubject_factor1_tensor = np.array(factor_dict["factor1"])
	Nsubject_factor2_tensor = np.array(factor_dict["factor2"])


	#save core, factors
	core_core, core_factor = decompose(Nsubject_core_tensor,rank = [Nsubject_core_tensor.shape[0],*core_compress])
	factor0_core, factor0_factor = decompose(Nsubject_factor0_tensor,rank = [Nsubject_core_tensor.shape[0],*factor0_compress])
	factor1_core, factor1_factor = decompose(Nsubject_factor1_tensor,rank = [Nsubject_core_tensor.shape[0],*factor1_compress])
	factor2_core, factor2_factor = decompose(Nsubject_factor2_tensor,rank = [Nsubject_core_tensor.shape[0],*factor2_compress])


	core_factor_T = [torch.from_numpy(x.T).to('cuda') for x in core_factor[1:]]
	factor0_factor_T  = [torch.from_numpy(x.T).to('cuda') for x in factor0_factor[1:]]
	factor1_factor_T  = [torch.from_numpy(x.T).to('cuda') for x in factor1_factor[1:]]
	factor2_factor_T  = [torch.from_numpy(x.T).to('cuda') for x in factor2_factor[1:]]
	with open('factor_all_subject.p', 'wb') as f:
	    pickle.dump((core_factor_T,factor0_factor_T,factor1_factor_T,factor2_factor_T), f)
	    


	for path in tqdm(combine_paths):
	    # print(path)
	    with open(path, 'rb') as f:
	        data = pickle.load(f)
	        core =  np.array(data[0])
	        factors = data[1]
	        
	        extract_core = tl.tenalg.multi_mode_dot(torch.from_numpy(core).to('cuda'),core_factor_T)
	        extract_factor0 = tl.tenalg.multi_mode_dot(torch.from_numpy(factors[0]).to('cuda'),factor0_factor_T)
	        extract_factor1 = tl.tenalg.multi_mode_dot(torch.from_numpy(factors[1]).to('cuda'),factor1_factor_T)
	        extract_factor2 = tl.tenalg.multi_mode_dot(torch.from_numpy(factors[2]).to('cuda'),factor2_factor_T)
	        
	        saveDir = os.path.join(os.path.dirname(path), 'feature_'+'_'.join(pair))
	        if not os.path.exists(saveDir):
	            os.mkdir(saveDir)
	        with open(os.path.join(saveDir, os.path.basename(path)),'wb') as f:
	            pickle.dump((extract_core.cpu().numpy(),extract_factor0.cpu().numpy(),extract_factor1.cpu().numpy(),extract_factor2.cpu().numpy()),f)
	        

	 #================================
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
	print(f"X.shape: {X.shape}, y.shape: {y.shape}")

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