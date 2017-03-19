# purpose of this repo
This repo is in charge of running machine learning algorithm and online classification assessment.

# the data for which this repo is designed
* the training samples in: https://github.com/birlrobotics/rcbht_data_processor/tree/master/my_training_data
* the streamed experiments in: https://github.com/birlrobotics/rcbht_data_processor/tree/master/my_streaming_experiments


# what can this repo do
1. train classifiers using training samples.  
    The following scripts are designed to train classifiers:
    * ./trainer_for_failure_classes.py
    * ./trainer_for_states.py
    * ./trainer_for_success_and_failure.py

    The models generated are stored in:
    * ./models_by_method

1. run online classification  
    The following scripts are desgiend to run online classification:
    * ./sim_online_classification.py

    You can config the online classification in this script:
    * ./config.py

    The result of online classificaion is in:
    * ./model_performance_in_online_classification


# related repo
The repo in charge of generating training samples and streaming experiments is https://github.com/birlrobotics/rcbht_data_processor

