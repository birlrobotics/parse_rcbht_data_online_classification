#!/usr/bin/env python

from sklearn.cross_validation import cross_val_score
from sklearn import svm, linear_model, naive_bayes, gaussian_process
from sklearn.externals import joblib
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import spline

import os

data_type = "SIM"
filepath = "/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data_cooker_code/parse_rcbht_data/my_training_data/"+data_type+"_HIRO_ONE_SA_SUCCESS"

all_mat = {}

filelist = ['training_set_for_approach', 'training_set_for_rotation',
    'training_set_for_insertion', 'training_set_for_mating']

for file in filelist:
    mat = np.genfromtxt(os.path.join(filepath, file), dtype='string', delimiter=',')
    #a half for training, a half for test
    all_mat[file] = mat

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "usage: %prog -m METHOD"
    parser = OptionParser(usage=usage)
    parser.add_option("-m", "--method",
        action="store", type="string", dest="method",
        help="training method, svm or logistic or gaussiannb")
    (options, args) = parser.parse_args()

    if options.method is None:
        parser.error("you have to provide a method in -m or --method")

    from trainer_common import CommonTrainer
    ct = CommonTrainer(all_mat, "data_type_"+data_type+"_model_for_state_", options.method)
    while True:
        ct.run_one_training()
