import os
import logging
from datetime import datetime,timedelta
from base.base_bobig import *
from base.tibero_dbconn import *
from base.query_sep import *
import inspect

def retrieve_name(var):
    #callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    #return [var_name for var_name,var_val in callers_local_vars if var_val is var]
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name,var_val in fi.frame.f_locals.items() if var_val is var] 
        if len(names) > 0:
            return names[0]

# def m_retrieve_name(var):
# callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
# return [var_name for var_name,var_val in callers_local_vars if var_val is var]    

def custom_logging(msg, printout=True, dbwrite=False, msg_code=""):
    msgVar = retrieve_name(msg)
    logging.info("{} : {}".format(msgVar, msg))
    if printout:
       print("{} : {}".format(msgVar, msg))
    if msg_code != "" and dbwrite:
       TBPBTV003_ins(msg_code, msg)

# 6. 배치상세내역 입력 함수
def TBPBTV003_ins(wk_exec_sts_cd, wk_exec_cnts):
    global BT_EXEC_SEQ,BT_SEQ
    TBPBTV003_ins_01_src = SQL_DIR + '/' + 'TBPBTV003_ins_01.sql'
    TBPBTV003_ins_01 = query_seperator(TBPBTV003_ins_01_src).format(bt_exec_seq=BT_EXEC_SEQ,
            bt_seq=BT_SEQ,
            wk_exec_sts_cd=wk_exec_sts_cd,
            wk_exec_cnts=wk_exec_cnts,
            crt_pgm_id=base_file_nm[0])
    custom_logging(TBPBTV003_ins_01)            
    cur.execute(TBPBTV003_ins_01)
    conn.commit()

WK_DTL_TP_CD = '7A'
BT_SEQ = 10

base_file_nm = os.path.basename(__file__).split('.')
logging.basicConfig(
    filename=LOG_DIR + '/' + base_file_nm[0]+ '_' + datetime.now().strftime('%Y%m%d')   + '.log', \
    level=eval(LOG_LEVEL), filemode='a+', \
    format='{} %(levelname)s : line = %(lineno)d , message = %(message)s'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

conn = tibero_db_conn()
cur = conn.cursor()
conn2 = tibero_db_conn2()
cur2 = conn2.cursor()

WK_DTL_TP_CD = '7A'
ASK_ID = ''
RSHP_ID = ''
EXEC_SEQ = 0
BT_SEQ = 0
BT_EXEC_SEQ = 0
CNORGCODE = ''

TBPBTV001_SEL_01_src = SQL_DIR + '/' + 'TBPBTV001_SEL_01.sql'
TBPBTV001_SEL_01 = query_seperator(TBPBTV001_SEL_01_src).format(wk_dtl_tp_cd=WK_DTL_TP_CD,
                                                            bt_seq =BT_SEQ,
                                                            crt_pgm_id = base_file_nm[0])
custom_logging(TBPBTV001_SEL_01)
cur.execute(TBPBTV001_SEL_01)
TBPBTV001_SEL_01_fetchall = cur.fetchall()
custom_logging(len(TBPBTV001_SEL_01_fetchall))
