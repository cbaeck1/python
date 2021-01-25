from base.base_bobig import *
import jaydebeapi

########## 개발 tibero DB ##########
TIBERO_DB_IP2 = '172.16.11.152'
TIBERO_DB_PORT2 = '8629'
TIBERO_DB_SID2 = 'hmbp_dev'
TIBERO_DB_USER2 = 'hmbpdata'
TIBERO_DB_PWD2 = '1234'

def tibero_db_conn2():

    conn = jaydebeapi.connect(
        TIBERO_JDBC_DRIVE,
        "jdbc:tibero:thin:@%s:%s:%s" % (TIBERO_DB_IP2 , TIBERO_DB_PORT2 , TIBERO_DB_SID2),
        [TIBERO_DB_USER2 , TIBERO_DB_PWD2],
        TIBERO_DB_JAR,
    )
    return conn

if __name__ == '__main__':
    print(tibero_db_conn2())


