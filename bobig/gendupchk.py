import hashlib

fsrc = open("dupchk.src", 'w')
fhash = open("dupchk.hash", 'w')
end = 10000000

for i in range(end):
    line = 'E3CD087091E9404A9DB075FF43CEB1A2' + str(i)
    fsrc.write(line +'\n')
    # print(line)
    encoded_string = line.encode()
    hexdigest = hashlib.sha256(encoded_string).hexdigest() + '\n'
    fhash.write(hexdigest.upper())

fsrc.close()
fhash.close()


'''
dupchk = pd.read_csv('dupchk.src')
duplicateRowsDF = dupchk[dupchk[0].duplicated()]
print(duplicateRowsDF)
'''


