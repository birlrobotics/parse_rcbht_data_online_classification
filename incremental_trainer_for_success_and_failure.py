#!/usr/bin/env python

from sklearn.cross_validation import cross_val_score
from sklearn import svm, linear_model, naive_bayes, gaussian_process
from sklearn.externals import joblib
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import spline

import os

from config import failure_class_name_to_id
data_type = "SIM"
failure_filepath = "/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data_cooker_code/parse_rcbht_data/my_training_data/"+data_type+"_HIRO_ONE_SA_ERROR_CHARAC_prob"
success_filepath = "/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data_cooker_code/parse_rcbht_data/my_training_data/"+data_type+"_HIRO_ONE_SA_SUCCESS"

all_mat = {}

filelist = [success_filepath+'/training_set_of_success', failure_filepath+'/training_set_of_fail']

min_amount = None

for file in filelist:
    mat = np.genfromtxt(file, dtype='string', delimiter=',')
    #a half for training, a half for test
    all_mat[file] = mat
 
    if min_amount is None or mat.shape[0] < min_amount:
        min_amount = mat.shape[0]  

for k in all_mat:
    np.random.shuffle(all_mat[k])
    all_mat[k] = all_mat[k][0:min_amount, :]


if __name__ == "__main__":
    from trainer_common import CommonTrainer
    ct = CommonTrainer(all_mat, "data_type_"+data_type+"_model_for_SF_", "")
#    while True:
#        ct.run_one_training()

    ct.run_incremental_trainings(graph_title="success/failure classification")
