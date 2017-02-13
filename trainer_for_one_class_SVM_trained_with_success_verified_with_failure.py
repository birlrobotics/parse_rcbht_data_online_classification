
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

succ_mat = np.genfromtxt(filelist[0], dtype='string', delimiter=',')
fail_mat = np.genfromtxt(filelist[1], dtype='string', delimiter=',')

if __name__ == "__main__":
    sample_size = succ_mat.shape[1]
    succ_amount = succ_mat.shape[0]


    succ_mat_for_train = succ_mat[:succ_amount/2, :]
    succ_mat_for_test = succ_mat[succ_amount/2:, :]

    succ_train_x = succ_mat_for_train[:, 0:sample_size-1]
    succ_train_y = succ_mat_for_train[:, sample_size-1:]

    succ_test_x = succ_mat_for_test[:, 0:sample_size-1]
    succ_test_y = succ_mat_for_test[:, sample_size-1:]

    fail_mat_for_test = fail_mat

    fail_test_x = fail_mat_for_test[:, 0:sample_size-1]
    fail_test_y = fail_mat_for_test[:, sample_size-1:]



    #best nu found: 0.1304    

    best = [0, 0, 0]
    nu = 0
    while nu < 1:
        nu += 0.0001

        model = svm.OneClassSVM(nu=nu).fit(succ_train_x)


        p_succ_train = model.predict(succ_train_x)
        p_succ_test = model.predict(succ_test_x)
        p_fail_test = model.predict(fail_test_x)

        succ_train_good = p_succ_train[p_succ_train == 1].size
        succ_test_good = p_succ_test[p_succ_test == 1].size
        fail_test_good = p_fail_test[p_fail_test == -1].size

        report = False
        if succ_train_good > best[0]:
            best[0] = succ_train_good
            report = True

        if succ_test_good > best[1]:
            best[1] = succ_test_good
            report = True

        if fail_test_good > best[2]:
            best[2] = fail_test_good
            report = True
    
        if report:
            print '---'
            print "nu", nu
            print "succ train:", succ_train_good, '/', p_succ_train.size
            print "succ test:", succ_test_good, '/', p_succ_test.size
            print "fail test:", fail_test_good, '/', p_fail_test.size
            

        
