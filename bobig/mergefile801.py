import os, sys, csv

'''
건보 폐쇄망 데이터 생성
key=/401/
  IF_DL_401_201800001_A0001_TBIPDV2018_1_1_20200531120000000.csv
2018-00001,A0001,K0001,96E7D6C9582304200C41D330302A80A6067F4451102AB7B940D808723C9F3B14,1
content=/501/

out=/

'''

print(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

key = '../2018/결합키재작업(최종)'+ sys.argv[1]
content = '../2018/기관데이터/'+sys.argv[3]
out = '../2018/폐쇄망/'+sys.argv[2] 
charset = sys.argv[4]

'''
key = 'IF_DL_401_201800001_A0001_TBIPDV2018_1_1_20200531120000000.csv'
content = 'IF_DL_501_201800001_A0001_TBIPDV2018_1_1_1_20200531120000000.txt'
out = 'IF_DL_801_201800001_A0001_TBIDIV2018_1_1_1_20200531120000000.txt'
'''

file_key = open(key, mode='rt', encoding='utf-8')
file_content = open(content, mode='rt', encoding=charset)
file_write = open(out, mode='wt', encoding='utf-8', newline='')

keyList1 = []
keyList2 = []
keyReader = csv.reader(file_key, delimiter=',')
for keyRow in keyReader:
  # 2018-00001,A0001,K0001,96E7D6C9582304200C41D330302A80A6067F4451102AB7B940D808723C9F3B14,1
  keyList1.append(keyRow[3])
  keyList2.append(keyRow[3:5])

writer = csv.writer(file_write, delimiter=',')
  
contentreader = csv.reader(file_content, delimiter=',')
first = 0
for row in contentreader:
  if first <= 7:
    if first == 3:
      row[3] = 'SEQ'
      writer.writerow(row)
    if first == 4:
      row[3] = '순번'
      writer.writerow(row)
    first = first + 1
    continue
 
  #print(row[3], keyList[0].index(row[3]))
  #row[3] = keyList[keyList.index(row[3])][1]
  #for keyRow in keyList1:
    #if keyRow[0] == row[3]:
    #  row[3] = keyRow[1]
    #  break
    #print(row[3], keyRow)
  #  print(row[3], keyRow, keyRow.index(row[3]))
  # print(row[3], keyList1.index(row[3]))
  #print(keyList2[keyList1.index(row[3])][1])

  row[3] = keyList2[keyList1.index(row[3])][1]
  writer.writerow(row)

file_key.close()
file_content.close()
file_write.close()

