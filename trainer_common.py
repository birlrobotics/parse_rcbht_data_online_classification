
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn import svm, linear_model, naive_bayes, gaussian_process
from sklearn.externals import joblib
import os

class CommonTrainer(object):
    def __init__(self, all_mat, custom_signature, train_method):
        self.all_mat = all_mat
        self.custom_signature = custom_signature
        self.train_method = train_method.lower()
        self.best_so_far = {}

    def run_incremental_trainings(self):
        import csv

        all_mat = self.all_mat
        train_method = self.train_method
        best_so_far = self.best_so_far


        test_set_samples = list()
        test_set_labels = list()


        # we always use the second half of data as test set
        for class_name in all_mat:
            mat = all_mat[class_name]
            np.random.shuffle(mat)

            max_num_train_sample = mat.shape[0]/2
            
            test_set_samples.append(mat[max_num_train_sample:, 0:len(mat[0]) - 1])
            test_set_labels.append(mat[max_num_train_sample:, len(mat[0]) - 1:])


        # we use the first half of data as training set in an incremental way 
        C = pow(10,-5)
        C_steps = 10
        size_steps = 10

        for now_kernel in ["linear", "poly", "rbf", "sigmoid", "precomputed"]:
            for now_C_step in range(C_steps):
                now_C = C*pow(10, now_C_step)
                for now_size_step in range(size_steps):
                    ratio = float(now_size_step)/size_steps

                    training_set_samples = list()
                    training_set_labels = list()

                    for class_name in all_mat:
                        mat = all_mat[class_name]

                        max_num_train_sample = mat.shape[0]/2
                        now_num_train_sample = int(ratio*max_num_train_sample)

                        #collect training samples 
                        training_set_samples.append(mat[0:now_num_train_sample, 0:len(mat[0]) - 1])
                        training_set_labels.append(mat[0:now_num_train_sample, len(mat[0]) - 1:])
                
                        from sklearn.model_selection import cross_val_score
                        clf = svm.SVC(kernel=now_kernel, C=now_C)
                        scores = cross_val_score(clf, iris.data, iris.target, cv=5)


                        print "now_kernel", now_kernel, "now_C", now_C, "now_num_train_sample", now_num_train_sample
                        print scores


    def run_one_training(self):
        all_mat = self.all_mat
        train_method = self.train_method
        best_so_far = self.best_so_far

        training_set_samples = list()
        training_set_labels = list()
        test_set_samples = list()
        test_set_labels = list()

        for class_name in all_mat:
            mat = all_mat[class_name]
            np.random.shuffle(mat)

            num_train_sample = mat.shape[0]/2
            print "num_train_sample", num_train_sample

            #collect training samples form the first half
            training_set_samples.append(mat[0:num_train_sample, 0:len(mat[0]) - 1])
            training_set_labels.append(mat[0:num_train_sample, len(mat[0]) - 1:])
            
            #coleect test samples from the other half
            test_set_samples.append(mat[num_train_sample:, 0:len(mat[0]) - 1])
            test_set_labels.append(mat[num_train_sample:, len(mat[0]) - 1:])

        #build matrix
        training_set_samples = np.array(np.vstack(training_set_samples), dtype=float)
        training_set_labels = np.array(np.vstack(training_set_labels), dtype=float)
        test_set_samples = np.array(np.vstack(test_set_samples), dtype=float)
        test_set_labels = np.array(np.vstack(test_set_labels), dtype=float)

        #train and get the model
        if train_method == 'svm':
            model = svm.SVC(decision_function_shape='ovo', probability=True).fit(training_set_samples, training_set_labels.ravel())
        elif train_method == 'logistic':
            model = linear_model.LogisticRegression().fit(training_set_samples, training_set_labels.ravel())
        elif train_method == 'gaussiannb':
            model = naive_bayes.GaussianNB().fit(training_set_samples, training_set_labels.ravel())
        else:
            raise Exception("method "+train_method+" is not supported now.")

        train_score = int(100*model.score(training_set_samples, training_set_labels.ravel()))
        print "train score:", train_score

        test_score = int(100*model.score(test_set_samples, test_set_labels.ravel()))
        print "test score:", test_score

        test_set_size = len(test_set_samples)

        predictions_for_test_set = model.predict(test_set_samples) 
        mask_of_correct_predictions = [int(test_set_labels[i][0] == predictions_for_test_set[i]) for i in range(test_set_size)]

        max_probas = [int(max(i)*100) for i in model.predict_proba(test_set_samples)]

        good_max_probas  = [max_probas[i]*mask_of_correct_predictions[i] for i in range(len(mask_of_correct_predictions))]

        avg_proba_by_class = {}

        for i in range(len(good_max_probas)):
            good_proba = good_max_probas[i]
            class_no = int(test_set_labels[i][0])
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
        result_dir_by_method = os.path.join("models_by_method", train_method)
        if not os.path.exists(result_dir_by_method):
            os.makedirs(result_dir_by_method)
        if new_progress: 
            signature = self.custom_signature 
            signature += "method_"+str(train_method)+"_"
            signature += "accuracy_"+str(test_score)+"_"
            signature += "avgProbaOfAll_"+str(avg_of_all)+"_"
            signature += "avgByClass_"
            for key, value in avg_proba_by_class.iteritems():
                signature += "C"+str(key)+"P"+str(value)+"_"
            from sklearn.externals import joblib
            joblib.dump(model, os.path.join(result_dir_by_method, signature+'.pkl'))
            f = open(os.path.join(result_dir_by_method, "detailed_proba_for_"+signature+".txt"), "w")
            f.write('\n'.join([str(test_set_labels[i][0])+" "+str(good_max_probas[i]) for i in range(len(good_max_probas))]))
            f.close()
