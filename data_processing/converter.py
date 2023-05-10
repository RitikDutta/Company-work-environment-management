import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
import pandas as pd
import csv
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import numpy as np
from sklearn.preprocessing import LabelEncoder
import base64
import cv2
import mediapipe as mp

class Converter:
    """
               This class shall be used for conversion of all Type of datatypes to files.

               Written By: Ritik Dutta
               Version: 1.0
               Revisions: None

    """

    def __init__(self):
        self.embedder = FaceNet()
        self.encoder = LabelEncoder()
        # self.path = path
        # self.class_name = class_name
        # self.landmarks = landmark_pb2.NormalizedLandmarkList()


    def convert_mp_to_csv(self, landmarks, class_name, path):
        """
                Method Name: convert_mp_to_csv
                Description: This method converts MediaPipe object to csv file for easy handling of data and training.
                Output: None
                On Failure: Raise Exception

                Written By: Ritik Dutta
                Version: 1.0
                Revisions: None

                        """
        try:
          # Open the CSV file for writing in append mode
            with open(path, 'a+', newline='') as csvfile:
                fieldnames = []
                for i in range(len(landmarks)):
                    fieldnames.extend([f"x{i+1}", f"y{i+1}", f"z{i+1}", f"visibility{i+1}"])
                # Add the class name as the last field
                fieldnames.append('class')
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		    	# Check if the file is empty
                csvfile.seek(0)
                first_char = csvfile.read(1)
                if not first_char:
                    # Write the field names as the first row of the CSV file
                    writer.writeheader()

                # Create a dictionary with the values of each NormalizedLandmark object in the list
                row = {}
                for i in range(len(landmarks)):
                    normalized_landmark = landmarks[i]
                    row[f"x{i+1}"] = normalized_landmark.x
                    row[f"y{i+1}"] = normalized_landmark.y
                    row[f"z{i+1}"] = normalized_landmark.z
                    row[f"visibility{i+1}"] = normalized_landmark.visibility
		    	# Add the class name to the row
                row['class'] = class_name
                # Write the dictionary as a row in the CSV file
                writer.writerow(row)
        except Exception as e:
            raise e


    def convert_mp_to_dataframe(self, landmarks):
        """
            Method Name: convert_mp_to_csv
            Description: This method converts MediaPipe object to pandas dataframe for easy handling of data and training.
            Output: None
            On Failure: Raise Exception

            Written By: Ritik Dutta
            Version: 1.0
            Revisions: None

                    """
        try:
            # Check None values
            if not landmarks or None in landmarks:
                return pd.DataFrame()
            
            # Create a dictionary with the values of each NormalizedLandmark object in the list
            row = {}
            for i in range(len(landmarks)):
                normalized_landmark = landmarks[i]
                row[f"x{i+1}"] = normalized_landmark.x
                row[f"y{i+1}"] = normalized_landmark.y
                row[f"z{i+1}"] = normalized_landmark.z
                row[f"visibility{i+1}"] = normalized_landmark.visibility
            
            # Create pandas dataframe from the dictionary
            df = pd.DataFrame([row])
            return df
        except Exception as e:
            raise e



    def convert_dict_to_dataframe(self, landmark_dict):
            """
                Method Name: convert_dict_to_dataframe
                Description: This method converts dictonary to pandas dataframe fetched from get request from flask webpage.
                Output: None
                On Failure: Raise Exception

                Written By: Ritik Dutta
                Version: 1.0
                Revisions: None

                        """
            try:
                x = landmark_dict
                columns = []
                data = []
                for i in range(len(x['landmarks'])):
                    columns.extend([f"x{i+1}", f"y{i+1}", f"z{i+1}", f"visibility{i+1}"])
                    data.extend([x['landmarks'][i]['x'], x['landmarks'][i]['y'], x['landmarks'][i]['z'], x['landmarks'][i]['visibility']])
                df = pd.DataFrame(data=[data], columns=columns)

                return df
            except Exception as e:
                raise e


    def convert_list_to_dataframe(self, landmarks):
        lst = landmarks[0][0]
        col_names = [f"{k}{i+1}" for i in range(len(lst)) for k in lst[i].keys()]
        values = [v for d in lst for v in d.values()]
        converted_dataframe = pd.DataFrame([values], columns=col_names)
        return converted_dataframe

    def convert_tuple_list_to_dataframe(self, landmarks):
        data = {}
        for i, landmark in enumerate(landmarks):
            data[f'x{i+1}'] = landmark[0]
            data[f'y{i+1}'] = landmark[1]
            data[f'z{i+1}'] = landmark[2]
            data[f'visibility{i+1}'] = landmark[3]
        converted_dataframe = pd.DataFrame(data, index=[0])
        print(converted_dataframe)
        return converted_dataframe

    def convert_json_to_face_image(self, data):

        img_base64 = data['image']
        img = base64.b64decode(img_base64.split(',')[1])
        npimg = np.frombuffer(img, np.uint8)
        face_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return face_image

    def get_landmarks(self, image):
        print(type(image))
        # image = cv2.imread(image)
        pose = mp.solutions.pose.Pose()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        landmarks = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                z = landmark.z
                landmarks.append((x, y, z))
        print(landmarks)
        return landmarks

    def get_embedding(self, image):
        image = image.astype('float32')
        image = np.expand_dims(image, axis=0)
        yhat = self.embedder.embeddings(image)
        return yhat[0]



    def append_npz_files(self,file_name, features, labels):
            try:
                with np.load(file_name) as data:
                    old_features = data['features']
                    old_labels = data['labels']
                    new_features = np.append(old_features, features, axis=0)
                    new_labels = np.append(old_labels, labels, axis=0)
            except FileNotFoundError:
                new_features = features
                new_labels = labels
            save_prompt = input("Do You want to updated Face Embeddings? ")
            if save_prompt == 'y':
                np.savez_compressed(file_name, features=new_features, labels=new_labels)
            else:
                pass

    def get_encoded(self, face_embeddings):
        y = face_embeddings['labels']
        self.encoder.fit(y)
        return y
