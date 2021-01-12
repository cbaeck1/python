import os, sys, csv

'''
암센터 : 데이터셋 위치 변경

'''

print(sys.argv[1] )

content = '../2018/기관데이터/503/'+sys.argv[1]
out = 'data/'+sys.argv[1]

'''
content = '../2018/기관데이터/503/'+'IF_DL_503_201800002_A0001_TBIPDV2018_1_1_20200608120000000.txt'
out = 'data/'+'out.new'
'''

file_content = open(content, mode='rt', encoding='utf-8')
file_write = open(out, mode='wt', encoding='utf-8', newline='')

writer = csv.writer(file_write, delimiter=',')
first = 0
contentreader = csv.reader(file_content, delimiter=',')
for row in contentreader:
    if first == 0:
        first = first + 1
        writer.writerow(row)
        continue

    # print(row[-1])
    row.insert(4, row[-1])
    #  리스트의 마지막은 제외
    row  = row[0:-1]
    writer.writerow(row)

file_content.close()
file_write.close()


'''
input_path = sys.argv[1]
input_header = sys.argv[2]
input_file = sys.argv[3]
output_file = sys.argv[4]

first_file = True
for input_file in glob.glob(os.path.join(input_path,'sales_*')):
  print(os.path.basename(input_file))
print(os.path.basename(input_header))
print(os.path.basename(input_file))

with open(input_header, 'r', newline='') as csv_in_header:
  headerreader = csv.reader(csv_in_header)      
  with open(input_header, 'r', newline='') as csv_in_file:
    filereader = csv.reader(csv_in_file)  
    with open(output_file, 'a', newline='') as csv_out_file:
      filewriter = csv.writer(csv_out_file)
      for row in headerreader:
        filewriter.writerow(row)
      for row in filereader:
        filewriter.writerow(row)

    if first_file:
      for row in filereader:
        filewriter.writerow(row)
      first_file = False
    else:
      header = next(filereader)
      for row in filereader:
        filewriter.writerow(row)
'''        