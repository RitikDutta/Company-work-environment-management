import datetime
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd
from IPython.display import display
from tabulate import tabulate
import os
import time
from data_processing.counter import counter
class CassandraCRUD:
    def __init__(self, keyspace):
        self.keyspace = keyspace
        cloud_config= {'secure_connect_bundle': 'secure-connect-test.zip'}
        auth_provider = PlainTextAuthProvider('biTEHxuyRqFgCcFpnEMOMvkN', 'PyqybfDQpPOaLp44hlWbb1Yb7bZ2Mn5Mt-_DGMOs.qBj.JY.Z,FAnB.9ncCD+F2EsSJ7W6XD,l,3S9gZW7bgWSMQDHKvTI7Iams2.hRRz6W_biRSrttvGgs1WhAl-in9')
        self.cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        self.session = self.cluster.connect(self.keyspace)
        self.counter = counter()

    def create_daily_activity_table(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS daily_activity (
            employee_id text,
            date date,
            start_time time,
            end_time time,
            work_hours text,
            phone_usage_duration text,
            looking_away_duration text,
            looking_away_frequency int,
            PRIMARY KEY (employee_id, date)
        );
        """
        self.session.execute(create_table_query)
        
    def create_total_activity_table(self):
        create_table = """
        CREATE TABLE IF NOT EXISTS total_activity (
            employee_id text,
            name text,
            total_work_duration text,
            total_phone_usage_duration text,
            total_looking_away_duration text,
            total_looking_away_frequency int,
            PRIMARY KEY (employee_id),
        );
        """
        self.session.execute(create_table)

        


    def insert_data(self, employee_id, date, start_time=datetime.datetime.now().time(), end_time=datetime.time(), work_hours="00:00:00", phone_usage_duration="00:00:00", looking_away_duration="00:00:00", looking_away_frequency=0, name="_", total_work_duration="00:00:00", total_phone_usage="00:00:00", total_looking_away_time="00:00:00", total_looking_away_frequency=0, new_employee=True):
        insert_query = """
        INSERT INTO daily_activity (
        employee_id, 
        date, 
        start_time,
        end_time,
        work_hours,
        phone_usage_duration, 
        looking_away_duration,
        looking_away_frequency
        ) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', {})""".format(employee_id, date, start_time, end_time, work_hours, phone_usage_duration, looking_away_duration, looking_away_frequency)
        self.session.execute(insert_query)
        self.show("daily_activity")
        
        select_query = "SELECT * FROM total_activity WHERE employee_id = %s"
        result = self.session.execute(select_query, (employee_id,)).one()
        if not result:
            #Insert into Total_activity
            insert_query = """
            INSERT INTO total_activity (
            employee_id, 
            name,
            total_work_duration,
            total_phone_usage_duration,
            total_looking_away_duration,
            total_looking_away_frequency
            ) VALUES ('{}', '{}', '{}', '{}', '{}', {})""".format(employee_id, "name", "00:00:00", "00:00:00", "00:00:00", 0)
            self.session.execute(insert_query)
            self.show("total_activity")

    def update_activity(self, employee_id, date, end_time=datetime.datetime.now().time(), work_hours=0, phone_usage_duration=0, looking_away_duration=0, looking_away_frequency=0, total_work_duration=0, total_phone_usage_duration=0, total_looking_away_duration=0, total_looking_away_frequency=0):
        # First, get the current values for the employee and date
        if not self.is_present(employee_id, date):
            print("-"*50)
            print("New Record Added")
            self.insert_data(employee_id, date)
            print("-"*50)

        select_query = "SELECT end_time, work_hours, phone_usage_duration, looking_away_duration, looking_away_frequency FROM daily_activity WHERE employee_id = %s and date = %s"
        daily_result = self.session.execute(select_query, (employee_id, date)).one()

        # Update the values with the new values
        t_end_time = end_time if daily_result else 0
        t_work_hours = self.counter.time_adder(daily_result.work_hours, work_hours) if daily_result else 0
        t_phone_usage_duration = self.counter.time_adder(daily_result.phone_usage_duration, phone_usage_duration) if daily_result else 0
        t_looking_away_duration = self.counter.time_adder(daily_result.looking_away_duration, looking_away_duration) if daily_result else 0
        t_looking_away_frequency = daily_result.looking_away_frequency + looking_away_frequency if daily_result else 0


        update_query = "UPDATE daily_activity SET end_time = %s, work_hours = %s, phone_usage_duration = %s, looking_away_duration = %s, looking_away_frequency = %s WHERE employee_id = %s and date = %s"
        self.session.execute(update_query, (t_end_time, t_work_hours, t_phone_usage_duration, t_looking_away_duration, t_looking_away_frequency, employee_id, date))
        # self.show("daily_activity")


        # Update contradicting values to total_activity table
        select_query = "SELECT total_work_duration, total_phone_usage_duration, total_looking_away_duration, total_looking_away_frequency FROM total_activity WHERE employee_id = %s"
        total_result = self.session.execute(select_query, (employee_id, )).one()

        # Update the values with the new values
        print(":: {}, {}".format(total_result.total_work_duration, work_hours))
        total_work_duration = self.counter.time_adder(total_result.total_work_duration, work_hours) if total_result else 0
        total_phone_usage_duration = self.counter.time_adder(total_result.total_phone_usage_duration, phone_usage_duration) if total_result else 0
        total_looking_away_duration = self.counter.time_adder(total_result.total_looking_away_duration, looking_away_duration) if total_result else 0
        total_looking_away_frequency += looking_away_frequency + total_result.total_looking_away_frequency if total_result else 0

        update_query = "UPDATE total_activity SET total_work_duration = %s, total_phone_usage_duration = %s, total_looking_away_duration = %s, total_looking_away_frequency = %s WHERE employee_id = %s"
        self.session.execute(update_query, (total_work_duration, total_phone_usage_duration, total_looking_away_duration, total_looking_away_frequency, employee_id))
        # self.show("total_activity")




    def select_data(self, table_name, condition):
        select_data_query = "SELECT * FROM " + table_name + " WHERE " + condition + ";"
        rows = self.session.execute(select_data_query)

        return rows

    def delete_data(self, table_name, condition):
        delete_data_query = "DELETE FROM " + table_name + " WHERE " + condition + ";"
        self.session.execute(delete_data_query)
        
    def delete_table(self, table_name):
        delete_table_query = "DROP TABLE " + table_name + ";"
        try:
            self.session.execute(delete_table_query)
            print("Table " + table_name + " deleted successfully")
        except Exception as e:
            print("Error deleting table: ", e)
            
    def show(self, table_name):
        query = "SELECT * FROM test_key.{};".format(table_name)
        df = pd.DataFrame(list(self.session.execute(query)))
        print(tabulate(df, headers='keys', tablefmt='fancy_grid'))

    def show_both(self):
        while True:
            self.show("daily_activity")
            self.show("total_activity")
            time.sleep(1)
            os.system("clear")
        
        
    def is_present(self, employee_id, date):
        select_query = "SELECT * FROM daily_activity WHERE employee_id = %s and date = %s"
        result = self.session.execute(select_query, (employee_id, date)).one()
        return True if result else False
    
    def get_db(self, table):
        query = "SELECT * FROM {}.{}".format(self.keyspace, table)
        data = self.session.execute(query)
        df = pd.DataFrame([d for d in data])
        return df
