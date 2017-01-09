#!/usr/bin/env python



import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import spline

import os

from config import failure_class_name_to_id
data_type = "SIM"
filepath = "/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data_cooker_code/parse_rcbht_data/my_training_data/"+data_type+"_HIRO_ONE_SA_ERROR_CHARAC_prob"


failure_class = failure_class_name_to_id.keys()
all_mat = {}

for fc in failure_class:
    file_name = "training_set_of_failure_class_"+fc
    try:
        mat = np.genfromtxt(os.path.join(filepath, file_name), dtype='string', delimiter=',')
    except IOError:
        print "no data for class", fc
        continue
    
    #a half for training, a half for test

    if len(mat.shape) == 1:
        mat = mat.reshape((1, mat.shape[0]))

    #shuffle all these trails before collecting them
    all_mat[fc] = mat







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
    ct = CommonTrainer(all_mat, "data_type_"+data_type+"_model_for_failure_class_", options.method)
    while True:
        ct.run_one_training()

