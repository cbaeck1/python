import csv

# content = 'D:\\data\\2018\\기관데이터\\501\\IF_DL_501_201800003_A0001_TBIPDV2018_4_1_1_20200531120000000.txt'
# content = 'D:\\data\\2018\\기관데이터\\502\\201800003_header\\IF_DL_502_201800003_A0001_TBIPDV2018_1_2_144_20200531120000000.txt'
content = 'D:\\data\\2018\\기관데이터\\502\\IF_DL_502_201800003_A0001_TBIPDV2018_4_1_1_20200531120000000.txt'
# content = 'D:\\data\\2018\\기관데이터\\503\\IF_DL_503_201800002_A0001_TBIPDV2018_1_1_20200608120000000.txt'
# content = 'D:\\data\\2018\\기관데이터\\IF_DL_504_201800001_A0001_TBIPDV2018_5_1_1_20200531120000000.txt'
# content = 'D:\\data\\2018\\기관데이터\\504\\IF_DL_504_201800001_A0002_TBIPDV2018_1_1_1_20200531120000000.txt'

# content = 'D:\\data\\2018\\기관데이터\\0612_BACKUP\\Result_DT3_17.txt'
# content = 'data/IF_DL_504_201800001_A0001_TBIPDV2018_2_1_1_20200531120000000.txt'

file_content = open(content, mode='rt', encoding='utf-8')

contentlist = []
contentreader = csv.reader(file_content, delimiter=',')
rowIndex = 0
errCount = 0
headerCount = 0
for row in contentreader:
  contentlist.append(len(row))
  if rowIndex >= 1 and rowIndex < 8:
    headerCount = contentlist[rowIndex]
    print(rowIndex, headerCount)
  if rowIndex >= 8:    
    #if contentlist[rowIndex] != 9:
      #print(errCount, rowIndex, contentlist[rowIndex-1], contentlist[rowIndex])
      if headerCount != contentlist[rowIndex]:
        errCount = errCount + 1
        print(errCount, rowIndex, contentlist[rowIndex-1], contentlist[rowIndex])
        #print(row)
  
  rowIndex = rowIndex + 1

print('total='+str(rowIndex), 'first='+str(contentlist[0]), 'header='+ str(contentlist[1]), 'errCount='+str(errCount))
#for row in contentlist:
#  print(row)

