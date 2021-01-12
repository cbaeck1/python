import os, sys

'''
심평원 : 헤더 + 데이터
'''

print(sys.argv[1], sys.argv[2] )

header = 'header/'+sys.argv[1]
content = '../2018/기관데이터/502/201800003/'+sys.argv[2]
out = 'data/'+sys.argv[2]


'''
header = 'syntax/'+'IF_DL_502_201800003_A0001_TBIPDV2018_1_1_144_20200531120000000.txt'
content = 'syntax/'+'IF_DL_502_201800003_A0001_TBIPDV2018_1_1_144_20200531120000000.txt'
out = 'syntax/'+'out.new'
'''

file_header = open(header, mode='rt', encoding='utf-8')
file_content = open(content, mode='rt', encoding='utf-8')
file_write = open(out, mode='wt', encoding='utf-8', newline='')
full_txt = ""
full_txt = file_header.read()
full_txt = full_txt +'\n' + file_content.read()

file_write.write(full_txt)

file_header.close()
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