import os, sys, csv

'''
장기기증과
기증자 전체중 결합키데이터만 추출

key = 'D:\\work\\2018\\결합개인식별정보\\2018-00008\\20200619\\'
IF_DL_401_201800008_A0001_TBIPDV2018_1_1_20200531120000000.txt'
content = 'D:\\work\\2018\\기관데이터\\504\\
    IF_DL_504_201800008_A0001_TBIPDV2018_1_1_1_20200531120000000.txt
    IF_DL_504_201800008_A0002_TBIPDV2018_1_1_1_20200531120000000.txt
out = 'data/'+sys.argv[2]
RSHP_ID = sys.argv[3]
'''

print(sys.argv[1], sys.argv[2], sys.argv[3])

key = 'D:\\work\\2018\\결합개인식별정보\\2018-00008\\20200619\\' + sys.argv[1]
content = 'D:\\work\\2018\\기관데이터\\504\\' + sys.argv[2] 
out = 'data/'+sys.argv[2]
RSHP_ID = sys.argv[3]


file_key = open(key, mode='rt', encoding='utf-8')
file_content = open(content, mode='rt', encoding='utf-8')
file_write = open(out, mode='wt', encoding='utf-8', newline='')

keyreader = csv.reader(file_key, delimiter=',')
keyList = []
first = 0
for readKeyRow in keyreader:
    if first == 0:
        first = first + 1
        continue

    if RSHP_ID == 'A0001':
        #print(readKeyRow[3])
        keyList.append(readKeyRow[3])
    else:
        #print(readKeyRow[0:3]+readKeyRow[4])
        keyList.append(readKeyRow[4])

writer = csv.writer(file_write, delimiter=',')
contentreader = csv.reader(file_content, delimiter=',')

contentList = []
first = 0
for readRow in contentreader:
    if first <= 7:
        first = first + 1
        contentList.append(readRow)
        #writer.writerow(readRow)
        continue
    
    for keyValue in keyList:
        if keyValue == readRow[3]:
            contentList.append(readRow)
            #writer.writerow(readRow)

# 중복제거
# newList = list(set(contentList))
new_list = []
first = 0
for v in contentList:
    if first <= 7:
        first = first + 1
        new_list.append(v)
        continue

    if v not in new_list:
        new_list.append(v)


for content in new_list:
    writer.writerow(content)

file_key.close()
file_content.close()
file_write.close()

