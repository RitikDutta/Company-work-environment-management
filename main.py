from data_collection.data_collector import Data_collection
import sys



try:
    sys.argv[1]
except IndexError:
    print('main.py has arguments:\nclass name:  name of class to capture data')
else:
	collection = Data_collection(sys.argv[1])
	collection.pose_collection()