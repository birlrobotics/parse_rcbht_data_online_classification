online_classification_data_type = "REAL"
arm_type = "ONE"
stream_interval = "0.1"

streaming_data_dir = '/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/parse_rcbht_data/my_streaming_experiments/stream_interval_%s/%s'%(stream_interval, online_classification_data_type)

label_map = {
    'approach' : 0,
    'rotation' : 1,
    'insertion' : 2,
    'mating' : 3,

    'success' : 1,
    'failure' : 0
}

proba_threshold = 0.7

model_dir = "./good_models"

if online_classification_data_type == "REAL":
    model_for_SF = "data_type_REAL_model_for_success_and_failure_method_svm_traincount_10114_accuracy_100_avgProbaOfAll_81_avgByClass_C0P79_C1P83_.pkl"
    model_for_states = "data_type_REAL_model_for_state_method_svm_traincount_79_accuracy_95_avgProbaOfAll_84_avgByClass_C0P89_C1P75_C2P85_C3P86_.pkl"
elif online_classification_data_type == "SIM":
    model_for_SF = "data_type_SIM_model_for_success_and_failure_method_svm_traincount_7966_accuracy_100_avgProbaOfAll_97_avgByClass_C0P98_C1P95_.pkl"
    model_for_states = "data_type_SIM_model_for_state_method_svm_traincount_2774_accuracy_89_avgProbaOfAll_59_avgByClass_C0P87_C1P79_C2P31_C3P39_.pkl"
    

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
