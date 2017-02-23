streaming_data_dir = '/Users/sklaw_mba/Desktop/ex/dr_juan_proj/workshop/data_cooker_code/parse_rcbht_data/my_streaming_experiments'

label_map = {
    'approach' : 0,
    'rotation' : 1,
    'insertion' : 2,
    'mating' : 3,
    'success' : 1,
    'failure' : 0
}

proba_threshold = 0.93

model_dir = "./good_models"
#model_for_SF = "data_type_SIM_model_for_success_and_failure_method_svm_accuracy_100_avgProbaOfAll_97_avgByClass_C0P98_C1P96_.pkl"
model_for_SF = "test.pkl"
model_for_states = "data_type_SIM_model_for_state_method_svm_accuracy_82_avgProbaOfAll_54_avgByClass_C0P86_C1P67_C2P16_C3P48_.pkl"

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
