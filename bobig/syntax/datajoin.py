import hashlib, os, csv

join = open("syntax/abc.hash", 'r', encoding='utf-8')
abc = csv.reader(join)

fr = open("syntax/csv.txt", 'r', encoding='utf-8')
fw = open("syntax/join.hash", 'w', encoding='utf-8', newline='')
rdr = csv.reader(fr)
wr = csv.writer(fw)
for line in rdr:
  # print(line)
  encoded_string = line[0].encode()
  line[0] = hashlib.sha256(encoded_string).hexdigest().upper()
  for joinline in abc:
    if joinline[0] == line[0]:
      wr.writerow(line)
      break
  
fr.close()
fw.close()
join.close()


