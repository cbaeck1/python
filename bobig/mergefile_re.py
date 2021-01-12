import os, sys, csv

print(sys.argv[1])

header = 'header/'+sys.argv[1]
content = '../2018/기관데이터/501/'+sys.argv[1]
out = 'data/'+sys.argv[1]

file_header = open(header, mode='rt', encoding='utf-8')
file_content = open(content, mode='rt', encoding='utf-8')
file_write = open(out, mode='wt', encoding='utf-8', newline='')

headerlist = []
headerreader = csv.reader(file_header)
for row in headerreader:
    headerlist.append(row)

writer = csv.writer(file_write, delimiter=',')
for row in headerlist:
    writer.writerow(row)

contentlist = []
contentreader = csv.reader(file_content, delimiter=',')
first = 0
for row in contentreader:
    if first <= 8:
        first = first + 1
        continue
    writer.writerow(row)

file_header.close()
file_content.close()
file_write.close()


