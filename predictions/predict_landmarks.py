import numpy as np
from data_processing.converter import Converter
import pandas as pd
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

class Prediction:
    """
               This class shall be used for predictions of landmarks of data.

               Written By: Ritik Dutta
               Version: 1.0
               Revisions: None

               """

    def __init__(self):
        self.convertor = Converter()
        # Load the model
        self.face_model = load_model('models/face_model.h5')
        # self.pose_model = load_model('models/pose_model.h5')


        # self.landmark = landmark_pb2.NormalizedLandmarkList()
    def predict(self, landmark):
        """
                    Method Name: predict
                    Description: This method predict the class from landmark data.
                    Output: None
                    On Failure: Raise Exception
    
                    Written By: Ritik Dutta
                    Version: 1.0
                    Revisions: None
    
        """
        if isinstance(landmark, dict):
            print("data frame")
            x = self.convertor.convert_dict_to_dataframe(landmark)
        else:
            print("Pose object")
            x = self.convertor.convert_mp_to_dataframe(landmark)
            x = pd.DataFrame(x.iloc[0:].values.reshape(1, -1))
            x = x.apply(pd.to_numeric, errors='coerce', downcast='float')
        print(x.shape)

        class_labels = {0: 'away', 1: 'phone', 2: 'working'}

        
    
        # Make a prediction
        prediction = self.pose_model.predict(x)
    
        # Find the index of the highest probability
        class_index = np.argmax(prediction)
    
        # Look up the class label in the dictionary
        class_label = class_labels[class_index]
    
        # Print the class label
    #     print("Class label:", class_label)
        return class_label 
    
    def predict_face(self, landmark):
        """
                    Method Name: predict
                    Description: This method predict the class from landmark data.
                    Output: None
                    On Failure: Raise Exception

                    Written By: Ritik Dutta
                    Version: 1.0
                    Revisions: None

        """
        if isinstance(landmark, dict):
            print("data frame")
            x = self.convertor.convert_dict_to_dataframe(landmark)
        else:
            print("Pose object")
            x = self.convertor.landmarks_to_df(landmark)
        print(x.shape)

        class_labels = {0: 'ritik', 1: 'rakshit'}
        # Make a prediction
        prediction = self.face_model.predict(x)
        print(max(prediction[0]))
        # Find the index of the highest probability
        class_index = np.argmax(prediction)
    
        # Look up the class label in the dictionary
        class_label = class_labels[class_index]
    
        # Print the class label
    #     print("Class label:", class_label)
        confidence = f"{class_label} {max(prediction[0]):.3f}"
        return confidence