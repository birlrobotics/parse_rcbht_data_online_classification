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
filepath = "/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data_cooker_code/parse_rcbht_data/training_data_of_failure_classes/"+data_type+"_HIRO_ONE_SA_ERROR_CHARAC_prob"

def main(method_name):
    global best_so_far
    global data_type 

    method_name = method_name.lower()

    train_x = list()
    train_y = list()
    test_x = list()
    test_y = list()

    failure_class = failure_class_name_to_id.keys()

    for fc in failure_class:
        mat = np.genfromtxt(os.path.join(filepath, fc), dtype='string', delimiter=',')
        #a half for training, a half for test
        num_train_sample = mat.shape[0]/2

        #shuffle all these trails before collecting them
        np.random.shuffle(mat)
        
        #collect training samples form the first half
        train_x.append(mat[0:num_train_sample, 0:len(mat[0]) - 1])
        train_y.append(mat[0:num_train_sample, len(mat[0]) - 1:])
        
        #coleect test samples from the other half
        test_x.append(mat[num_train_sample:, 0:len(mat[0]) - 1])
        test_y.append(mat[num_train_sample:, len(mat[0]) - 1:])

    #build matrix
    train_x = np.array(np.vstack(train_x), dtype=float)
    train_y = np.array(np.vstack(train_y), dtype=float)
    test_x = np.array(np.vstack(test_x), dtype=float)
    test_y = np.array(np.vstack(test_y), dtype=float)



    #train and get the model
    if method_name == 'svm':
        model = svm.SVC(decision_function_shape='ovo', probability=True).fit(train_x, train_y.ravel())
    elif method_name == 'logistic':
        model = linear_model.LogisticRegression().fit(train_x, train_y.ravel())
    elif method_name == 'gaussiannb':
        model = naive_bayes.GaussianNB().fit(train_x, train_y.ravel())
    else:
        raise Exception("method "+method_name+" is not supported now.")

    train_score = int(100*model.score(train_x, train_y.ravel()))
    print "train score:", train_score

    test_score = int(100*model.score(test_x, test_y.ravel()))
    print "test score:", test_score

    predict_answer = model.predict(test_x) 
    good_counts = [int(test_y[i][0] == predict_answer[i]) for i in range(len(predict_answer))]

    max_probas = [int(max(i)*100) for i in model.predict_proba(test_x)]

    good_max_probas  = [max_probas[i]*good_counts[i] for i in range(len(good_counts))]

    avg_proba_by_class = {}

    for i in range(len(good_max_probas)):
        good_proba = good_max_probas[i]
        class_no = int(test_y[i][0])
        if class_no not in avg_proba_by_class:
            avg_proba_by_class[class_no] = []
        avg_proba_by_class[class_no].append(good_proba)

    
    new_progress = False 
    
    if "accuracy" not in best_so_far:
        best_so_far["accuracy"] = 0
    if test_score > best_so_far["accuracy"]:
        best_so_far["accuracy"] = test_score
        new_progress =True

    if "all" not in best_so_far:
        best_so_far["all"] = 0
    avg_of_all = sum(good_max_probas)/len(good_max_probas)
    if avg_of_all > best_so_far["all"]:
        best_so_far["all"] = avg_of_all
        new_progress = True

    for class_no in avg_proba_by_class:
        if class_no not in best_so_far:
            best_so_far[class_no] = 0
        avg_proba_by_class[class_no] = sum(avg_proba_by_class[class_no])/len(avg_proba_by_class[class_no])    
        if avg_proba_by_class[class_no] > best_so_far[class_no]:
            best_so_far[class_no] = avg_proba_by_class[class_no]
            new_progress = True


    print "avg_of_all", avg_of_all
    print "avg_proba_by_class", avg_proba_by_class
    result_dir_by_method = os.path.join("models_by_method", method_name)
    if not os.path.exists(result_dir_by_method):
        os.makedirs(result_dir_by_method)
    if new_progress: 
        signature = ""
        signature += "dataType_"+data_type+"_"
        signature += "method_"+str(method_name)+"_"
        signature += "accuracy_"+str(test_score)+"_"
        signature += "avgProbaOfAll_"+str(avg_of_all)+"_"
        signature += "avgByClass_"
        for key, value in avg_proba_by_class.iteritems():
            signature += "C"+str(key)+"P"+str(value)+"_"
        from sklearn.externals import joblib
        joblib.dump(model, os.path.join(result_dir_by_method, 'model_for_failure_class_classification_'+signature+'.pkl'))
        f = open(os.path.join(result_dir_by_method, "detailed_proba_for_"+signature+".txt"), "w")
        f.write('\n'.join([str(test_y[i][0])+" "+str(good_max_probas[i]) for i in range(len(good_max_probas))]))
        f.close()

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

    while True:
        main(options.method);

