import hashlib, os, csv

fr = open("syntax/tab.txt", 'r', encoding='utf-8')
fw = open("syntax/tab.hash", 'w', encoding='utf-8', newline='')
rdr = csv.reader(fr, delimiter='\t')
wr = csv.writer(fw, delimiter='\t')
for line in rdr:
  print(line)
  encoded_string = line[0].encode()
  line[0] = hashlib.sha256(encoded_string).hexdigest().upper()
  wr.writerow(line)

fr.close()
fw.close()

