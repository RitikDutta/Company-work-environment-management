from data_processing.counter import counter
from database.database_operations import CassandraCRUD
# from predictions import live_predict
import datetime
class database_handler():
    def __init__(self):
        self.countert = counter()
        self.db = CassandraCRUD("test_key")
        # self.predictions = live_predict()
    
    def df_handle(self, action, name):
        updated, action, duration, pre_action, pre_duration, pre_name = self.countert.action_tracker(action, name)
        print("--", pre_duration)
        d = datetime.datetime.today().date()
        t = datetime.datetime.today().time()
        if name:
            print("insert {} for {} into {} record".format(pre_action, pre_duration, pre_name))
            if pre_action == "phone":
                self.db.update_activity(employee_id=str(pre_name), date=d, phone_usage_duration=pre_duration, end_time=t)
            elif pre_action == "working":
                self.db.update_activity(employee_id=str(pre_name), date=d, work_hours=pre_duration, end_time=t)
            elif pre_action == "looking away":
                self.db.update_activity(employee_id=str(pre_name), date=d, looking_away_duration=pre_duration, end_time=t)



    # def get_prediction(self):
    #     face, pose = self.predictions.show_both()
    #     print(face, pose)