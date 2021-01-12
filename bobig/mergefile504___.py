import os, sys, csv

'''
사용하지 않음
header = 'data/'+sys.argv[1]
content = 'header/'+sys.argv[2]
out = 'data/'+sys.argv[3]
2018-00001	DT4 A0001
2018-00003	DT3 A0001
2018-00004	DT5	A0001 A0002
'''
ASK_ID = '2018-00004'
file_id = 'DT5'
RSHP_ID = 'A0002'

PRVDR_CD = 'K0004'
headerfilelist = ['1','1','1','2','2','2','3','3','3','4','4']
outfilelist = ['1','2','3','4','5','6','7','8','9','10','11']
filelist = ['07','08','09','10','11','12','13','14','15','16','17']
datasetlist = ['HN07_ALL','HN08_ALL','HN09_ALL','HN10_ALL','HN11_ALL','HN12_ALL','HN13_ALL','HN14_ALL','HN15_ALL','HN16_ALL','HN17_ALL']

for i in range(11):
    # print(headerfilelist[i], type(headerfilelist[i]))
    header = 'header/'+'IF_DL_504'+ '_'+ ASK_ID.replace('-','') + '_' + RSHP_ID + '_TBIPDV2018_' + headerfilelist[i] + '_1_1_20200531120000000.txt'
    dataset = datasetlist[i]
    if RSHP_ID == 'A0001':
        content = 'data/'+'Result_' + file_id + '_' + filelist[i] + '.txt' 
    else:
        content = 'data/'+'Result2_' + file_id + '_' + filelist[i] + '.txt'    
    out = 'data/'+'IF_DL_504'+ '_'+ ASK_ID.replace('-','') + '_' + RSHP_ID + '_TBIPDV2018_' + outfilelist[i] + '_1_1_20200531120000000.txt'

    file_header = open(header, mode='rt', encoding='utf-8')
    file_write = open(out, mode='wt', encoding='utf-8', newline='')
    file_content = open(content, mode='rt', encoding='utf-8')

    headerlist = []
    headerreader = csv.reader(file_header)
    for row in headerreader:
        headerlist.append(row)
    #print(headerlist)

    writer = csv.writer(file_write, delimiter=',')
    for row in headerlist:
        writer.writerow(row)

    contentlist = []
    contentreader = csv.reader(file_content, delimiter='\t')
    first = 0
    for readRow in contentreader:
        if first == 0:
            first = first + 1
            continue
        newRow = []
        for column in readRow:
            column = column.replace(","," ")   
            newRow.append(column)

        newRow.insert(0, ASK_ID)
        newRow.insert(1, RSHP_ID)
        newRow.insert(2, PRVDR_CD)
        newRow.insert(4, dataset)
        contentlist.append(newRow)
    # print(contentlist)

    for row in contentlist:
        writer.writerow(row)
    
    file_header.close()
    file_write.close()
    file_content.close()

'''
file_write = open(out, mode='wt', encoding='utf-8')
full_txt = ""
full_txt = file_header.read()
full_txt = full_txt + file_content.read()

file_write.write(full_txt)


'''

