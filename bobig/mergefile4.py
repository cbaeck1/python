import os, sys, csv

'''
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
outfilelist = ['1','2','3','4','5','6','7','8','9','10','11']
filelist = ['07','08','09','10','11','12','13','14','15','16','17']
datasetlist = ['HN07_ALL','HN08_ALL','HN09_ALL','HN10_ALL','HN11_ALL','HN12_ALL','HN13_ALL','HN14_ALL','HN15_ALL','HN16_ALL','HN17_ALL']

for i in range(11):
    # print(headerfilelist[i], type(headerfilelist[i]))
    dataset = datasetlist[i]
    if RSHP_ID == 'A0001':
        content = 'data/'+'Result_' + file_id + '_' + filelist[i] + '.txt' 
    else:
        content = 'data/'+'Result2_' + file_id + '_' + filelist[i] + '.txt'    
    out = 'data/'+'IF_DL_504'+ '_'+ ASK_ID.replace('-','') + '_' + RSHP_ID + '_TBIPDV2018_' + outfilelist[i] + '_1_1_20200531120000000.txt'

    file_write = open(out, mode='wt', encoding='utf-8', newline='')
    file_content = open(content, mode='rt', encoding='utf-8')

    headerlist = [['TBIPDV2018','1','1']]
    writer = csv.writer(file_write, delimiter=',')
    for row in headerlist:
        writer.writerow(row)

    contentlist = []
    contentreader = csv.reader(file_content, delimiter='\t')
    first = 0
    for row in contentreader:
        if first == 0:
            first = first + 1
            headerYn = ['N','N','N','N','N']
            headerTableName =  ['TBIPDV2018','TBIPDV2018','TBIPDV2018','TBIPDV2018','TBIPDV2018']
            headerColumName =  ['ASK_ID','RSHP_ID','PRVDR_CD','HASH_DID','DATASET']
            headerAttrName =  ['신청_ID','연구가설_ID','제공기관코드','해시DID','데이터셋']
            headerDatatype =  ['문자','문자','문자','문자','문자']
            headerDatalen =  ['10','5','5','64','200']
            headerDescription =  ['신청_ID','연구가설_ID','제공기관코드','해시DID','데이터셋']
            fisrtCol = 0
            for rowStr in row:
                # first column == n_jumin7
                if fisrtCol == 0:
                    fisrtCol = fisrtCol + 1
                    continue
                headerYn.append('N')
                headerTableName.append(dataset)
                headerColumName.append(rowStr)
                headerAttrName.append(rowStr)
                headerDatatype.append('문자')
                headerDatalen.append('4000')
                headerDescription.append(rowStr)

            contentlist.append(headerYn)
            contentlist.append(headerTableName)
            contentlist.append(headerColumName)
            contentlist.append(headerAttrName)
            contentlist.append(headerDatatype)
            contentlist.append(headerDatalen)
            contentlist.append(headerDescription)
            continue

        newRow = []
        for column in row:
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
    
    file_write.close()
    file_content.close()

'''
file_write = open(out, mode='wt', encoding='utf-8')
full_txt = ""
full_txt = file_header.read()
full_txt = full_txt + file_content.read()

file_write.write(full_txt)


'''

