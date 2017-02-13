
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

    import random
    from datetime import datetime
    random.seed(datetime.now())
    while True:
        model = svm.OneClassSVM(random_state=random.randint(0, 271462243)).fit(succ_train_x)
        print succ_amount
        print "---"
        print model.predict(succ_train_x)
        print model.predict(succ_test_x)

    
