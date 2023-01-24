from data_collection.data_collector import Data_collection
from predictions.live_predict import LivePredict
from src.train_model import ModelTraining
import sys


def data_collection():
    try:
        collection = Data_collection(sys.argv[2], sys.argv[3])
        collection.holistic_collection()
    except IndexError:
        print('main.py has arguments:\nclass name:  name of class to capture data')


def test_prediction():
    prediction = LivePredict()
    prediction.live_predict(sys.argv[2])


def train_face():
    try:
        training = ModelTraining()
        training.train_model(n_components=int(sys.argv[2]), epochs=int(sys.argv[3]), input_model='models/face_model.h5')
    except IndexError:
        print("please give the following parameters train_model(n_components, epochs)")


if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except KeyError as e:
        print("An KeyError occurred:", e)
    except IndexError:
        print("please enter the operation name")
        print("data_collection()\ntest_prediction()\ntrain_face()")
