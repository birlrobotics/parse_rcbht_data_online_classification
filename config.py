streaming_data_dir = '/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data_cooker_code/parse_rcbht_data/my_streaming_experiments'

label_map = {
    'approach' : 0,
    'rotation' : 1,
    'insertion' : 2,
    'mating' : 3,
    'success' : 1,
    'failure' : 0
}

proba_threshold = 0.95

model_dir = "./models_by_method"
model_for_states = "svm/model_for_state_classification_dataType_REAL_method_svm_accuracy_92_avgProbaOfAll_73_avgByClass_C0P88_C1P72_C2P67_C3P64_.pkl"
