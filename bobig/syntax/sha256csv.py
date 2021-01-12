import hashlib, os, csv

fr = open("syntax/csv.txt", 'r', encoding='utf-8')
fw = open("syntax/csv.hash", 'w', encoding='utf-8', newline='')
rdr = csv.reader(fr)
wr = csv.writer(fw)
for line in rdr:
  print(line)
  encoded_string = line[0].encode()
  line[0] = hashlib.sha256(encoded_string).hexdigest().upper()
  wr.writerow(line)
  
fr.close()
fw.close()


'''
# 첫번째 컬럼을 key값으로 사용
with open('syntax/csv.txt', encoding='utf-8') as csvfile:
  rdr = csv.DictReader(csvfile)
  for i in rdr:
    print(i)

with open('names.csv', 'w', newline='') as csvfile:
    fieldnames = ['first_name', 'last_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
    writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
    writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})    
'''

'''
# using pandas
import numpy as pd

train = pd.read_csv("syntax/csv.txt")
# train 데이터 살펴보기
train.describe(include="all")
'''






