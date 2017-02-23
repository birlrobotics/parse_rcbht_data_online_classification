import config
import os
from sklearn.externals import joblib
import numpy
from PIL import Image

color_plate = [
    (255 , 255 ,255),
    (255, 0 ,0),
    (0, 255, 0)
]
    

print os.listdir(config.streaming_data_dir)

proba_threshold = config.proba_threshold

proba_threshold = config.proba_threshold

while proba_threshold < 1:
    for class_dir in os.listdir(config.streaming_data_dir):
        class_name = class_dir
        if class_name[0] == '.':
            continue
        right_answer = config.label_map[class_name]
        class_no = right_answer

        if class_name in ["success", "failure"]:
            model_name = config.model_for_SF
            model_signature = model_name.split('.')[0]
        elif class_name in ["approach", "rotation", "insertion", "mating"]:
            continue
            model_name = config.model_for_states
            model_signature = model_name.split('.')[0]
        else:
            raise Exception("I don't have a model for class "+class_name+"!")

        print class_dir, right_answer

        streams_prediction = []
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

            if len(one_stream_prediction) > max_len:
                max_len = len(one_stream_prediction)

            streams_prediction.append(one_stream_prediction)

        
        output_pixels = []
        for i in streams_prediction:
            l = len(i) 
            output_pixels += [color_plate[j] for j in i]
            output_pixels += [(0, 0, 0)]*(max_len-l)

        img_height = len(streams_prediction)
        img_width = max_len
        
        output_signature = config.online_classification_data_type+"_"+model_signature+"_thre_"+str(proba_threshold)+"_" 
        output_dir = os.path.join("model_performance_in_online_classification", output_signature)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_img = Image.new("RGB", (img_width, img_height)) # mode,(width,height)
        output_img.putdata(output_pixels)
        zoom = 1
        output_img = output_img.resize((img_width*zoom, img_height*zoom))
        output_img.save(os.path.join(output_dir, str(class_no)+"_"+class_name+"_online_performance.png"))
        output_img.save(os.path.join(output_dir, str(class_no)+"_"+class_name+"_online_performance.eps"))
    proba_threshold += 0.05
