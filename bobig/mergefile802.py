import os, sys, csv

'''
건보 폐쇄망 데이터 생성을 위한 키생성
key=/401/
  IF_DL_401_201800001_A0001_TBIPDV2018_1_1_20200531120000000.csv
2018-00001,A0001,K0001,96E7D6C9582304200C41D330302A80A6067F4451102AB7B940D808723C9F3B14,1
content=/501/
  IF_DL_501_201800008_B0001_TBIPDV2018_1_1_1_20200531120000000.txt

'''

print(sys.argv[1], sys.argv[2], sys.argv[2])

key = '../2018/결합키재작업(최종)'+ sys.argv[1]
content = '../2018/기관데이터/'+sys.argv[2]
content2 = '../2018/기관데이터/'+sys.argv[3]
'''
key = 'IF_DL_401_201800008_B0001_TBIPDV2018_1_1_20200531120000000.csv'
content = 'IF_DL_501_201800008_B0001_TBIPDV2018_1_1_1_20200531120000000.txt'
content = 'IF_DL_501_201800008_B0001_TBIPDV2018_9_1_1_20200531120000000.txt'
'''

file_content = open(content, mode='rt', encoding='cp949')
file_content2 = open(content2, mode='rt', encoding='cp949')
file_key = open(key, mode='wt', encoding='utf-8', newline='')

# 2018-00008,B0001,K0001,4A78C5D3A001F3D5FDA6797354927ED5D921F8C873C50BB5122C2A8D7DCB1F59,BFC,NA,2009,2,45,7,27230,0,2,2
contentreader = csv.reader(file_content, delimiter=',')
first = 0
contentList = []
for row in contentreader:
  if first <= 7:
    first = first + 1
    continue

  if first == 8:
    ASK_ID=row[0]
    RSHP_ID=row[1]
    PRVDR_CD=row[2]

  contentList.append(row[3])

contentreader2 = csv.reader(file_content2, delimiter=',')
first = 0
for row in contentreader2:
  if first <= 7:
    first = first + 1
    continue

  contentList.append(row[3])
  

# 중복제거
# newList = list(set(contentList))
new_list = []
iRow = 0
for v in contentList:
    if v not in new_list:
        #print(iRow, v)
        iRow = iRow + 1
        new_list.append(v)


# 1 부터 시작
iRow = 1
key_list = []
for v in new_list:
    nList = []
    nList.append(ASK_ID)
    nList.append(RSHP_ID)
    nList.append(PRVDR_CD)
    nList.append(v)
    nList.append(iRow)
    #print(iRow, nList)
    iRow = iRow + 1
    key_list.append(nList)

writer = csv.writer(file_key, delimiter=',')
for content in key_list:
    writer.writerow(content)

file_key.close()
file_content.close()

