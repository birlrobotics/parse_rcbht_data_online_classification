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

failure_x_base = ["+x", ""]
failure_y_base = ["+y", "-y", ""]
failure_r_base = ["+r", "-r", ""]

failure_class_name_to_id = {}
id_count = 0
for x in failure_x_base:
    for y in failure_y_base:
        for r in failure_r_base:
            now_class = x+y+r
            failure_class_name_to_id[now_class] = id_count
            id_count += 1
