import hashlib

fr = open("abc.txt", 'r', encoding='UTF8')
fw = open("abc.hash", 'w', encoding='UTF8')
while True:
    lines = fr.readlines(10000)
    if not lines: break
    for line in lines:
        line = ''.join(list(map(lambda s: s.strip(), line)))
        # print(line)
        encoded_string = line.encode()
        hexdigest = hashlib.sha256(encoded_string).hexdigest() + '\n'
        fw.write(hexdigest.upper())

fr.close()
fw.close()
