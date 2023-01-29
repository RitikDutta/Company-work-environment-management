import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
import pandas as pd
import csv
import tensorflow as tf
from joblib import load


class Converter:
    """
               This class shall be used for conversion of all Type of datatypes to files.

               Written By: Ritik Dutta
               Version: 1.0
               Revisions: None

    """

    def __init__(self):
        # self.face_pca = load('models/face_pca.joblib')
        # self.class_name = class_name
        # self.landmarks = landmark_pb2.NormalizedLandmarkList()
        pass
    def convert_mp_to_csv(self, landmarks, landmark_type, class_name, path):
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
            # Save landmarks to csv
            if getattr(landmarks, landmark_type) is not None:
                if landmark_type == "multi_face_landmarks":
                    landmarks_data = getattr(landmarks, landmark_type)[0].landmark
                else:
                    landmarks_data = getattr(landmarks, landmark_type).landmark

                num_of_landmarks = len(landmarks_data)
            else:
                landmarks_data = None
                num_of_landmarks = 21
            with open(path + '_' + landmark_type + '.csv', 'a+', newline='') as csvfile:
                fieldnames = []
                for i in range(num_of_landmarks):
                    fieldnames.extend([f"x{i + 1}", f"y{i + 1}", f"z{i + 1}"])
                    if landmark_type == "pose_landmarks":
                        fieldnames.extend([f"visibility{i + 1}"])
                fieldnames.append('class')
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                csvfile.seek(0)
                first_char = csvfile.read(1)
                if not first_char:
                    writer.writeheader()
                row = {}
                if landmarks_data is not None:
                    for i in range(num_of_landmarks):
                        normalized_landmark = landmarks_data[i]
                        row[f"x{i + 1}"] = normalized_landmark.x
                        row[f"y{i + 1}"] = normalized_landmark.y
                        row[f"z{i + 1}"] = normalized_landmark.z
                        if landmark_type == "pose_landmarks":
                            row[f"visibility{i + 1}"] = normalized_landmark.visibility
                else:
                    for i in range(num_of_landmarks):
                        row[f"x{i + 1}"] = 0
                        row[f"y{i + 1}"] = 0
                        row[f"z{i + 1}"] = 0
                row['class'] = class_name
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
            # if not landmarks or None in landmarks:
            #     return pd.DataFrame()

            # Create a dictionary with the values of each NormalizedLandmark object in the list
            row = {}
            for i in range(len(landmarks)):
                normalized_landmark = landmarks[i]
                row[f"x{i + 1}"] = normalized_landmark.x
                row[f"y{i + 1}"] = normalized_landmark.y
                row[f"z{i + 1}"] = normalized_landmark.z
                row[f"visibility{i + 1}"] = normalized_landmark.visibility

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
                columns.extend([f"x{i + 1}", f"y{i + 1}", f"z{i + 1}", f"visibility{i + 1}"])
                data.extend([x['landmarks'][i]['x'], x['landmarks'][i]['y'], x['landmarks'][i]['z'],
                             x['landmarks'][i]['visibility']])
            df = pd.DataFrame(data=[data], columns=columns)

            return df
        except Exception as e:
            raise e

    def landmarks_to_df(self, landmarks):
        face_pca = load('models/face_pca.joblib')
        landmarks = landmarks.landmark
        data = []
        row = {}
        for i, landmark in enumerate(landmarks):
            row[f"x{i + 1}"] = landmark.x
            row[f"y{i + 1}"] = landmark.y
            row[f"z{i + 1}"] = landmark.z
        data.append(row)
        df = pd.DataFrame(data)
        df = face_pca.transform(df)
        return df
