import pandas as pd
import ora1

##-- SQL query
query = """
SELECT topic, count(*) as cnt
    FROM help
    WHERE 1 = 1
GROUP BY topic
ORDER BY cnt
               """

##-- Excute OracleDB SQL in Python
result = ora1.query_OracleSQL(query)

print(result)


