 INSERT INTO TBPBTV003 
(BT_DTL_SEQ,
 BT_EXEC_SEQ,
 BT_SEQ,
 WK_EXEC_ST_DT,
 WK_EXEC_END_DT,
 WK_EXEC_STS_CD,
 WK_EXEC_CNTS,
 CRT_PGM_ID,
 CRT_DT)
 VALUES
 (TBPBTV003_BT_DTL_SEQ.NEXTVAL,
{bt_exec_seq},
{bt_seq},
sysdate,
sysdate,
'{wk_exec_sts_cd}',
'{wk_exec_cnts}',
'{crt_pgm_id}',
sysdate)
