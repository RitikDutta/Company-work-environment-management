import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from data_collection.data_collector import Data_collection
from predictions.live_predict import LivePredict
from src.train_model import ModelTraining
import sys


def data_collection():
    try:
        collection = Data_collection(class_name=sys.argv[2], collection_type=sys.argv[3])
        
        # collection.pose_collection()
        collection.face_collection()
    except IndexError:
        print('main.py has arguments:\nclass name:  name of class to capture data')


def test_prediction():
    try:
        prediction = LivePredict()
        # if sys.argv[2] == "pose":
        #     prediction.show_pose()
        # elif sys.argv[2] == "face":
        #     prediction.show_face(sys.argv[3])
        prediction.show_both()
    except IndexError:
        print("Please Select from\n1.Pose\n2.Face")



def train():
    try:
        training = ModelTraining()
        if sys.argv[2] == "face":
            training.train_face_model(data_directory="models/faces_embeddings.npz",
                                 model_output_directory="models/faces_embeddings.npz",
                                 # model_input_directory='models/face_model.h5',
                                 )
        elif sys.argv[2] == "pose":
            training.train_model(data_directory="raw_data/training/landmarks.csv",
                                 keras_model_output_directory="models/pose_model.h5",
                                 # keras_model_input_directory='models/pose_model.h5',
                                 # pca_model_input_directory="models/pca_model.joblib",
                                 pca_model_output_directory="models/pose_pca.joblib",
                                 n_components=15)

    except IndexError:
        print("please give the following parameters train_model(n_components, epochs)")


if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except KeyError as e:
        print("An KeyError occurred:", e)
    except IndexError:
        print("please enter the operation name")
        print("data_collection(class_name, collection_type)\ntest_prediction()\ntrain_face()")
