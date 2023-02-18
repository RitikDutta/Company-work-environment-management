import datetime
class counter():
    def __init__(self):
        self.init_time = datetime.datetime.now()
        self.previous_time = 0
        self.pre_action = ""
        self.prev_action = ""
        self.prev_name = ""
        self.pre_name = ""
        self.last_recorded = self.init_time
        self.time_eta = datetime.timedelta(seconds=0)
    def is_updated(self, action, name):
        if (action == self.prev_action) and (name == self.prev_name):
            print("Not updated")
            return False
        else:
            print("updated")
            self.prev_action = action
            self.prev_name = name
            return True
    
    def action_tracker(self, action, name):
        if self.is_updated(action, name):
            is_updated = True
            self.init_time = datetime.datetime.now()
            self.previous_time = self.time_eta.seconds + (self.init_time - self.last_recorded).seconds
            pre_action = self.pre_action
            pre_name = self.pre_name
            pre_duration = datetime.timedelta(seconds=self.previous_time)
            print("previous action {} for {}".format(self.pre_action, datetime.timedelta(seconds=self.previous_time)))    
            self.pre_action = action
            self.pre_name = name
        else:
            is_updated = False
            pre_action = ""
            pre_name = ""
            pre_duration = datetime.timedelta(seconds=0)
        self.time_eta = datetime.timedelta(seconds=(datetime.datetime.now() - self.init_time).seconds)
        self.last_recorded = datetime.datetime.now()
        cur_action = action
        cur_duration = self.time_eta
        print("{} {} since {}".format(name, action, self.time_eta))
        
        if self.time_eta.seconds > 10:
            print("nice")
            
        return is_updated, cur_action, cur_duration, pre_action, pre_duration.seconds, pre_name
    
    def time_adder(self, str_time, addition):
            print(str_time, addition)
            date_select = datetime.datetime.strptime(str_time, '%H:%M:%S')
            delta = datetime.timedelta(seconds=addition)
            target_time = date_select + delta
            return str(target_time.strftime("%H:%M:%S"))
    