import os, sys, csv

dataset = '200'

file_content = open('./syntax/a1.txt', mode='rt', encoding='utf-8')
file_write = open('./syntax/a1.out', mode='wt', encoding='utf-8', newline='')
writer = csv.writer(file_write, delimiter=',')

contentlist = []
contentreader = csv.reader(file_content, delimiter=',')

for readRow in contentreader:
    print(readRow[0])
    print(readRow[0][0:4]+'-'+readRow[0][4:9])
    #print(readRow[0][9:14])
    #print(readRow[0][14:19])
    #print(readRow[0][19:83])
    #print(readRow[1:])

    newRow = []
    newRow.append(readRow[0][0:4]+'-'+readRow[0][4:9])
    newRow.append(readRow[0][9:14])
    newRow.append(readRow[0][14:19])
    newRow.append(readRow[0][19:83])
    newRow.append(dataset)
    #  리스트의 마지막은 제외
    newRow  = newRow + readRow[1:-1]
    print(newRow)
    contentlist.append(readRow)

for writeRow in contentlist:
    writer.writerow(writeRow)

file_write.close()
file_content.close()
