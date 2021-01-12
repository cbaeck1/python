import csv

contentA = 'C:/data/test_data_a.txt'
contentB = 'C:/data/test_data_b.txt'
out = 'C:/data/test_combine_ab.txt'


file_contentA = open(contentA, mode='rt', encoding='utf-8')
file_contentB = open(contentB, mode='rt', encoding='utf-8')

headerlist = []
contentAlist = []
firstA = 0
contentAReader = csv.reader(file_contentA)
for rowA in contentAReader:
  if firstA == 0:
    firstA = firstA + 1
    headerlist = rowA
    continue   
  # contentAlist.append(rowA[0])
  newRow = []
  for column in rowA:
    newRow.append(column)
  contentAlist.append(newRow)

print("A", contentAlist[0])
# print("A", contentAlist[1][0])

file_write = open(out, mode='wt', encoding='utf-8', newline='')
writer = csv.writer(file_write, delimiter=',')

firstB = 0
contentBReader = csv.reader(file_contentB)
for rowB in contentBReader:
  if firstB == 0:
    firstB = firstB + 1
    headerlist = headerlist + rowB[1:]
    writer.writerow(headerlist)
    continue  
  
  '''  
  if rowB[0] in contentAlist:
    #print(rowB[1:])
    writer.writerow(rowB)
  '''
  
  for rowA in contentAlist:
    if rowB[0] in rowA[0]:
      #print(rowA+rowB[1:])
      writer.writerow(rowA+rowB[1:])

file_write.close()
file_contentA.close()
file_contentB.close()