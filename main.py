from data_collection.data_collector import Data_collection
from live_predict import LivePredict
import sys


def data_collection():
	try:
		collection = Data_collection(sys.argv[2])
		collection.pose_collection()
	except IndexError as e:
		print('main.py has arguments:\nclass name:  name of class to capture data')

def test_prediction():
	perdiction = LivePredict()
	perdiction.live_predict()

if __name__ == '__main__':
	try:
		globals()[sys.argv[1]]()
	except KeyError as e:
		print("An KeyError occurred:", e)