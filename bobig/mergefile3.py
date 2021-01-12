import os, sys, csv

'''

header = 'data/'+sys.argv[1]
content = 'header/'+sys.argv[2]
out = 'data/'+sys.argv[3]
IF_DL_504_201800001_A0002_TBIPDV2018_1800001_1_1_20200531120000000.txt
IF_DL_504_201800001_A0002_TBIPDV2018_1_1_1_20200531120000000.txt

'''
ASK_ID = '2018-00001'
PRVDR_CD = 'K0004'
file_id = 'data/IF_DL_504_201800001_A0002_TBIPDV2018_1800001_1_1_20200531120000000.txt'
RSHP_ID = 'A0002'
dataset = 'PBHL_BIG_DATA'

# print(headerfilelist[i], type(headerfilelist[i]))
out = 'data/IF_DL_504_201800001_A0002_TBIPDV2018_1_1_1_20200531120000000.txt'

file_write = open(out, mode='wt', encoding='utf-8', newline='')
file_content = open(file_id, mode='rt', encoding='cp949')

writer = csv.writer(file_write, delimiter=',')

# contentlist = []
first = 1
contentreader = csv.reader(file_content, delimiter=',')
for row in contentreader:
    if first <= 8:
        first = first + 1
    else:
        row.insert(4, dataset)
        if len(row) == 10:
            row[7] = row[7]+' ' +row[8]
            row[8] = row[9]
            del row[9]
            print(row)
    writer.writerow(row)
    #contentlist.append(row)
# print(contentlist)

file_write.close()
file_content.close()

'''
file_write = open(out, mode='wt', encoding='utf-8')
full_txt = ""
full_txt = file_header.read()
full_txt = full_txt + file_content.read()

file_write.write(full_txt)


'''

