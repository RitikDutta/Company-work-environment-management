import datetime
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd
class CassandraCRUD:
    def __init__(self, keyspace):
        cloud_config= {'secure_connect_bundle': 'secure-connect-test.zip'}
        auth_provider = PlainTextAuthProvider('biTEHxuyRqFgCcFpnEMOMvkN', 'PyqybfDQpPOaLp44hlWbb1Yb7bZ2Mn5Mt-_DGMOs.qBj.JY.Z,FAnB.9ncCD+F2EsSJ7W6XD,l,3S9gZW7bgWSMQDHKvTI7Iams2.hRRz6W_biRSrttvGgs1WhAl-in9')
        self.cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        self.session = self.cluster.connect(keyspace)

    def create_daily_activity_table(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS daily_activity (
            employee_id text,
            date date,
            start_time time,
            end_time time,
            work_hours float,
            phone_usage_duration float,
            looking_away_duration float,
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
            total_work_duration float,
            total_phone_usage_duration float,
            total_looking_away_duration float,
            total_looking_away_frequency int,
            PRIMARY KEY (employee_id),
        );
        """
        self.session.execute(create_table)

        


    def insert_data(self, employee_id, date, start_time=datetime.time(), end_time=datetime.time(), work_hours=0.0, phone_usage_duration=0.0, looking_away_duration=0.0, looking_away_frequency=0, name="_", total_work_duration=0, total_phone_usage=0, total_looking_away_time=0, total_looking_away_frequency=0):
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
        ) VALUES ('{}', '{}', '{}', '{}', {}, {}, {}, {})""".format(employee_id, date, start_time, end_time, work_hours, phone_usage_duration, looking_away_duration, looking_away_frequency)
        self.session.execute(insert_query)
        self.show("daily_activity")
        
        #Insert into Total_activity
        insert_query = """
        INSERT INTO total_activity (
        employee_id, 
        name,
        total_work_duration,
        total_phone_usage_duration,
        total_looking_away_duration,
        total_looking_away_frequency
        ) VALUES ('{}', '{}', {}, {}, {}, {})""".format(employee_id, "name", 0, 0, 0, 0)
        self.session.execute(insert_query)
        self.show("total_activity")

    def update_activity(self, employee_id, date, end_time, work_hours=0, phone_usage_duration=0, looking_away_duration=0, looking_away_frequency=0, total_work_duration=0, total_phone_usage=0, total_looking_away_time=0, total_looking_away_frequency=0):
        # First, get the current values for the employee and date
        select_query = "SELECT end_time, work_hours, phone_usage_duration, looking_away_duration, looking_away_frequency FROM daily_activity WHERE employee_id = %s and date = %s"
        result = self.session.execute(select_query, (employee_id, date)).one()

        # Update the values with the new values
        end_time = end_time if result else 0
        work_hours += result.work_hours if result else 0
        phone_usage_duration += result.phone_usage_duration if result else 0
        looking_away_duration += result.looking_away_duration if result else 0
        looking_away_frequency += result.looking_away_frequency if result else 0


        update_query = "UPDATE daily_activity SET end_time = %s, work_hours = %s, phone_usage_duration = %s, looking_away_duration = %s, looking_away_frequency = %s WHERE employee_id = %s and date = %s"
        self.session.execute(update_query, (end_time, work_hours, phone_usage_duration, looking_away_duration, looking_away_frequency, employee_id, date))
        self.show("daily_activity")
        
        
        # Update contradicting values to total_activity table
        select_query = "SELECT total_work_duration, total_phone_usage_duration, total_looking_away_duration, total_looking_away_frequency FROM total_activity WHERE employee_id = %s"
#         result = self.session.execute(select_query, (employee_id)).one()

        # Update the values with the new values
        total_work_duration += result.work_hours if result else 0
        total_phone_usage += result.phone_usage_duration if result else 0
        total_looking_away_time += result.looking_away_duration if result else 0
        total_looking_away_frequency += result.looking_away_frequency if result else 0

        update_query = "UPDATE total_activity SET total_work_duration = %s, total_phone_usage_duration = %s, total_looking_away_duration = %s, total_looking_away_frequency = %s WHERE employee_id = %s"
        self.session.execute(update_query, (total_work_duration, total_phone_usage, total_looking_away_time, total_looking_away_frequency, employee_id))
        self.show("total_activity")




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
        df = pd.DataFrame(list(crud.session.execute(query)))
        display(df) 
