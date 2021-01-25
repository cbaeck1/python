import os, sys
import logging
from datetime import datetime,timedelta
from base.base_bobig import *
from base.tibero_dbconn import *
from base.tibero_dbconn2 import *
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

# hmbpuser
conn = tibero_db_conn()
cur = conn.cursor()
# hmbpdata
conn2 = tibero_db_conn2()
cur2 = conn2.cursor()

ASK_ID = '2019-00008' # sys.argv[1]
RSHP_ID = 'A0001' # sys.argv[2]
PRVDR_CD = 'K0004' # sys.argv[3]

# 비식별기준이 없으면 파일 생성을 하지 않는다. 비식별만 재작업하기 위하여
PASS_NOT_EXISTS_TBPINV112 = False  # default = True
PASS_ALTER_ID = True # default = True

STDOUT_TRUE = True # default
STDOUT_FALSE = False
TBPBTV003_TRUE = True
TBPBTV003_FALSE = False # default

#################################################################
TBIPKV2019_SEL_01_src = SQL_DIR + '/' + 'TBIPKV2019_SEL_01.sql'
TBIPKV2019_SEL_01 = query_seperator(TBIPKV2019_SEL_01_src).format(ask_id=ASK_ID,
                                                            rshp_id=RSHP_ID,
                                                            prvdr_cd=PRVDR_CD)
custom_logging(TBIPKV2019_SEL_01)
cur2.execute(TBIPKV2019_SEL_01)
TBIPKV2019_SEL_01_fetchall = cur2.fetchall()
custom_logging(len(TBIPKV2019_SEL_01_fetchall))

TBPPKV2019_DEL_01_src = SQL_DIR + '/' + 'TBPPKV2019_DEL_01.sql'
TBPPKV2019_DEL_01 = query_seperator(TBPPKV2019_DEL_01_src).format(ask_id_num=ASK_ID[5:],
                                                            rshp_id=RSHP_ID,
                                                            prvdr_cd=PRVDR_CD)
custom_logging(TBPPKV2019_DEL_01)            
cur.execute(TBPPKV2019_DEL_01)

for TBIPKV2019_val_cnt, BIPKV2019_val in enumerate(TBIPKV2019_SEL_01_fetchall):
    TBPPKV2019_INS_01_src = SQL_DIR + '/' + 'TBPPKV2019_INS_01.sql'
    TBPPKV2019_INS_01 = query_seperator(TBPPKV2019_INS_01_src).format(ask_id=ASK_ID,
                                                                rshp_id=RSHP_ID,
                                                                prvdr_cd=PRVDR_CD,
                                                                hash_did=BIPKV2019_val[0],
                                                                crt_pgm_id=base_file_nm[0])
    #custom_logging(TBPPKV2019_INS_01)            
    cur.execute(TBPPKV2019_INS_01)

conn.commit()

