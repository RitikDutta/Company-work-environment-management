import numpy as np
from data_processing.converter import Converter
import pandas as pd
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
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
        self.facenet = FaceNet()
        self.face_embeddings = np.load("models/faces_embeddings.npz")
        self.class_labels = {0: 'away', 1: 'phone', 2: 'working'}
        # Load the model
        self.model = pickle.load(open("models/xgb_pose.pkl", 'rb'))
        # self.model = load_model('models/pose.h5')
        # Load the face model
        self.face_model = pickle.load(open("models/face_SVC_model.pkl", 'rb'))
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load('models/face_encoder.npy')

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

        d = ["{}{}".format(s, i) for i in range(1, 34) for s in ["x", "y", "z", "visibility"]]        
        x.columns = d
        # _ = [x.drop([f'x{i}', f'y{i}', f'z{i}', f'visibility{i}'], axis=1, inplace=True) for i in range(24, 34)]
    
        # Make a prediction
        prediction = self.model.predict(x)
    
        # Find the index of the highest probability
        class_index = np.argmax(prediction)
    
        # Look up the class label in the dictionary
        # class_label = self.class_labels[class_index]
    
        # Print the class label
    #     print("Class label:", class_label)

        # return str(prediction[0])
        print(prediction)
        return str(self.class_labels[prediction[0]])
        # return str(np.argmax(prediction))
        # if np.argmax(prediction) == 0:
        #     return("away")
        # elif np.argmax(prediction) == 1:
        #     return("phone")
        # elif np.argmax(prediction) == 2:
        #     return("working")


    def face_predict(self, img):
        y_pred = self.facenet.embeddings(img)
        face_name = self.face_model.predict(y_pred)
        final_name = self.encoder.inverse_transform(face_name)[0]
        return final_name

    def predict_df(self, landmark):
        prediction = self.model.predict(landmark)
        return str(self.class_labels[prediction[0]])