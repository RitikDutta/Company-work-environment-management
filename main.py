from data_collection.data_collector import Data_collection
from predictions.live_predict import LivePredict
import sys


def data_collection():
	try:
		collection = Data_collection(sys.argv[2], sys.argv[3])
		collection.holistic_collection()
	except IndexError as e:
		print('main.py has arguments:\nclass name:  name of class to capture data')

def test_prediction():
	perdiction = LivePredict()
	perdiction.live_predict(sys.argv[2])

if __name__ == '__main__':
	try:
		globals()[sys.argv[1]]()
	except KeyError as e:
		print("An KeyError occurred:", e)
	except IndexError:
		print("please enter the operation name")
		print("data_collection()\ntest_prediction()")