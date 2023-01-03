import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
import pandas as pd
import csv

class Converter:
    """
               This class shall be used for conversion of all Type of datatypes to files.

               Written By: Ritik Dutta
               Version: 1.0
               Revisions: None

    """

    def __init__(self):
        # self.path = path
        # self.class_name = class_name
        self.landmarks = landmark_pb2.NormalizedLandmarkList()


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













