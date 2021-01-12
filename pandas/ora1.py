# 1. Oracle DB에 Python으로 접속하여 SQL query 해서 pandas DataFrame 만들기

def query_OracleSQL(query):
     
     import pandas as pd
     import cx_Oracle as co
     from datetime import datetime

     start_tm = datetime.now()

     #  DB Connecion
     dsn_tns = co.makedsn("localhost", "1521", service_name="ORCL")
     conn = co.connect(user="system", password="admin", dsn=dsn_tns)

     # Get a dataframe
     query_result = pd.read_sql(query, conn)

     # Close connection
     conn.close()

     end_tm = datetime.now()
     print('START: ', str(start_tm))
     print('END: ', str(end_tm))
     print('ELAP: ', str(end_tm - start_tm))


     return query_result
 

 


