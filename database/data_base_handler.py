from data_processing.counter import counter
from database import database_operations
# from predictions import live_predict

class database_handler():
    def __init__(self):
        self.countert = counter()
        # self.predictions = live_predict()
    
    def df_handle(self, action, name):
        updated, action, duration, pre_action, pre_duration, pre_name = self.countert.action_tracker(action, name)
        print("insert {} for {} into {} record".format(pre_action, pre_duration, pre_name))

    # def get_prediction(self):
    #     face, pose = self.predictions.show_both()
    #     print(face, pose)