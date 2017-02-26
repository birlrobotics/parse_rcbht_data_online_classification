import config
import os
from sklearn.externals import joblib
import numpy
from PIL import Image

color_plate = [
    (255 , 255 ,255),
    (255, 0 ,0),
    (0, 255, 0),
    (0, 0, 255),
]
    

print os.listdir(config.streaming_data_dir)

proba_threshold = config.proba_threshold

green_count_start_at = float(2)/3


rows = []


while proba_threshold < 1:

    for class_dir in os.listdir(config.streaming_data_dir):


        class_name = class_dir
        if class_name[0] == '.':
            continue



        right_answer = config.label_map[class_name]
        class_no = right_answer

        if class_name in ["success", "failure"]:
            classification_type = "SF"
            model_name = config.model_for_SF
        elif class_name in ["approach", "rotation", "insertion", "mating"]:
            classification_type = "State"
            model_name = config.model_for_states

        else:
            raise Exception("I don't have a model for class "+class_name+"!")

        model_signature = model_name.split('.')[0]
        print class_dir, right_answer

        streams_prediction = []
        green_count_group_by_exp = []
        max_len = 0
        for exp in os.listdir(os.path.join(config.streaming_data_dir, class_dir)):
            if exp[0] == '.':
                continue
            one_stream_prediction = []
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

                max_proba = max(class_with_proba)
                predicted_y = class_with_proba.index(max_proba)


                if max_proba > proba_threshold: 
                    if int(predicted_y) == right_answer:
                        to_push = 2 
                    else:
                        to_push = 1 
                else :
                    to_push = 0
                one_stream_prediction.append(to_push)

            now_exp_length = len(one_stream_prediction)        

            if now_exp_length > max_len:
                max_len = now_exp_length

            green_count = 0
            

            for i in range(int(now_exp_length*green_count_start_at)+1, now_exp_length):
                if one_stream_prediction[i] == 2:
                    one_stream_prediction[i] = 3
                    green_count += 1            

            streams_prediction.append(one_stream_prediction)
            green_count_group_by_exp.append(green_count)

        
        output_pixels = []
        for i in streams_prediction:
            l = len(i) 
            output_pixels += [color_plate[j] for j in i]
            output_pixels += [(0, 0, 0)]*(max_len-l)

        img_height = len(streams_prediction)
        img_width = max_len
        
        output_signature = config.online_classification_data_type+"_"+model_signature+"_thre_"+str(proba_threshold)+"_" 
        output_dir = os.path.join("model_performance_in_online_classification", "stream_interval_"+config.stream_interval, config.online_classification_data_type, classification_type, output_signature)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_img = Image.new("RGB", (img_width, img_height)) # mode,(width,height)
        output_img.putdata(output_pixels)
        zoom = 1
        output_img = output_img.resize((img_width*zoom, img_height*zoom))
        output_img.save(os.path.join(output_dir, str(class_no)+"_"+class_name+"_online_performance.png"))
        output_img.save(os.path.join(output_dir, str(class_no)+"_"+class_name+"_online_performance.eps"))

        
        green_count_report = open(os.path.join(output_dir, str(class_no)+"_"+class_name+"_online_performance.txt"), "w")
        green_count_report.write(str(green_count_group_by_exp))
        green_count_report.close()

        import numpy


        list_of_exp_length = [len(i) for i in streams_prediction]
        avg_exp_length = numpy.mean(list_of_exp_length)

        now_dict = {}
        now_dict["data_type"] = config.online_classification_data_type
        now_dict["threshold"] = proba_threshold
        now_dict["class"] = class_name 
        now_dict["green_count_start_at"] = green_count_start_at
        now_dict["min_green_count"] = min(green_count_group_by_exp) 
        now_dict["mean_green_count"] = numpy.mean(green_count_group_by_exp)
        now_dict["max_green_count"] = max(green_count_group_by_exp) 
        now_dict["avg_trial_length"] = avg_exp_length


        rows.append(now_dict)

    proba_threshold += 0.05

import csv
fieldnames = ["data_type", 'threshold', 'class', 'green_count_start_at', 
    'min_green_count', 'mean_green_count', 'max_green_count', "avg_trial_length"]
csv_file = open(os.path.join("model_performance_in_online_classification", "stream_interval_"+config.stream_interval, "green_count_report_of_%s_data.txt"%(config.online_classification_data_type,)), "w")
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()
for row in rows:
    writer.writerow(row)
