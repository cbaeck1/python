import hashlib, os

fr = open("syntax/abc.txt", 'r', encoding='UTF8')
fw = open("syntax/abc.hash", 'w')
while True:
    line = fr.readline()
    print(line)
    line = ''.join(list(map(lambda s: s.strip(), line)))
    print(line)
    if not line: break
    encoded_string = line.encode()
    hexdigest = hashlib.sha256(encoded_string).hexdigest()
    fw.write(hexdigest.upper())
    fw.write('\n')

fr.close()
fw.close()
