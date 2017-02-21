#!/usr/bin/env python

from sklearn.cross_validation import cross_val_score
from sklearn import svm, linear_model, naive_bayes, gaussian_process
from sklearn.externals import joblib
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import spline

import os

data_type = "REAL"
arm_amount = "ONE"
filepath = "/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/parse_rcbht_data/my_training_data/"+data_type+"_HIRO_"+arm_amount+"_SA_SUCCESS"

all_mat = {}

filelist = ['training_set_for_approach', 'training_set_for_rotation',
    'training_set_for_insertion', 'training_set_for_mating']

for file in filelist:
    mat = np.genfromtxt(os.path.join(filepath, file), dtype='string', delimiter=',')
    #a half for training, a half for test
    all_mat[file] = mat

if __name__ == "__main__":
    from trainer_common import CommonTrainer
    ct = CommonTrainer(all_mat, "data_type_"+data_type+"_model_for_state_", "")
#    while True:
#        ct.run_one_training()

    ct.run_incremental_trainings(graph_title="state classification of %s-arm %s data"%(arm_amount , data_type))
