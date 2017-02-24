import config
import os
from sklearn.externals import joblib
import numpy
from PIL import Image

def plot_SF(output_dir, list_of_proba, title):
    x = []
    y_for_success = []
    y_for_failure = []
    now_x = 0
    for array in list_of_proba:
        x.append(now_x)
        now_x += 1
        y_for_success.append(array[1])
        y_for_failure.append(array[0])

    import matplotlib.pyplot as plt
    plt.plot(x, y_for_success, 'go-', label="success")
    plt.plot(x, y_for_failure, 'bo-', label="failure")

    plt.ylabel('probability')
    plt.xlabel('time splice')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(output_dir, title+".png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, title+".eps"), bbox_inches='tight')
    plt.clf()

    dat_file = open(os.path.join(output_dir, title+".txt"), "w")
    dat_file.write("x: "+str(x)+"\n")
    dat_file.write("y_for_success: "+str(y_for_success)+"\n")
    dat_file.write("y_for_failure: "+str(y_for_failure)+"\n")
    dat_file.close()

def plot_state(output_dir, list_of_state_proba, title):
    x = []
    y_for_approach = []
    y_for_rotation = []
    y_for_insertion = []
    y_for_mating = []
    now_x = 0
    for state in ["approach", "rotation", "insertion", "mating"]:
        for array in list_of_state_proba[state]:
            x.append(now_x)
            now_x += 1
            y_for_approach.append(array[0])
            y_for_rotation.append(array[1])
            y_for_insertion.append(array[2])
            y_for_mating.append(array[3])

    import matplotlib.pyplot as plt
    plt.plot(x, y_for_approach, 'ro-', label="approach state")
    plt.plot(x, y_for_rotation, 'go-', label="rotation state")
    plt.plot(x, y_for_insertion, 'bo-', label="insertion state")
    plt.plot(x, y_for_mating, 'ko-', label="mating state")

    plt.ylabel('probability')
    plt.xlabel('time splice')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(output_dir, title+".png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, title+".eps"), bbox_inches='tight')
    plt.clf()

    dat_file = open(os.path.join(output_dir, title+".txt"), "w")
    dat_file.write("x: "+str(x)+"\n")
    dat_file.write("y_for_approach: "+str(y_for_approach)+"\n")
    dat_file.write("y_for_rotation: "+str(y_for_rotation)+"\n")
    dat_file.write("y_for_insertion: "+str(y_for_insertion)+"\n")
    dat_file.write("y_for_mating: "+str(y_for_mating)+"\n")
    dat_file.close()


def main():
    print os.listdir(config.streaming_data_dir)

    result_dict = {}

    for class_dir in os.listdir(config.streaming_data_dir):
        class_name = class_dir
        if class_name[0] == '.':
            continue

        if class_name in ["success", "failure"]:
            model_name = config.model_for_SF
            model_signature = model_name.split('.')[0]
        elif class_name in ["approach", "rotation", "insertion", "mating"]:
            model_name = config.model_for_states
            model_signature = model_name.split('.')[0]
        else:
            raise Exception("I don't have a model for class "+class_name+"!")

        for exp in os.listdir(os.path.join(config.streaming_data_dir, class_dir)):
            if exp[0] == '.':
                continue

            if exp not in result_dict:
                result_dict[exp] = {}

            result_dict[exp][class_name] = []

            file_of_one_stream = open(\
                os.path.join(\
                    config.streaming_data_dir,\
                    class_dir,\
                    exp,\
                    "file_of_one_stream.txt"\
                )\
            )

            for line in file_of_one_stream:
                model = joblib.load(os.path.join(config.model_dir, model_name))
                line = line.strip().strip(',')
                x = [int(i) for i in line.split(',')]
                x = numpy.array(x).reshape((1, -1))
                class_with_proba = model.predict_proba(x)[0].tolist()
                result_dict[exp][class_name].append(class_with_proba)

    output_dir = os.path.join("prob_graph", config.arm_type+" arm", config.online_classification_data_type+" data", "success trials")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join("prob_graph", config.arm_type+" arm", config.online_classification_data_type+" data", "failure trials" )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for exp in result_dict:
        if "success" in result_dict[exp]:
            output_dir = os.path.join("prob_graph", config.arm_type+" arm", config.online_classification_data_type+" data", "success trials")
            plot_SF(output_dir, result_dict[exp]["success"], config.online_classification_data_type+" data SF probability graph of trial "+exp)
            plot_state(output_dir, result_dict[exp], config.online_classification_data_type+" data state probability graph of trial "+exp)
        elif "failure" in result_dict[exp]:
            output_dir = os.path.join("prob_graph", config.arm_type+" arm", config.online_classification_data_type+" data", "failure trials")
            plot_SF(output_dir, result_dict[exp]["failure"], config.online_classification_data_type+" data SF probability graph of trial "+exp)



main()
