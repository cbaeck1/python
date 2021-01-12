
fsrc = open("dupchk.hash", 'r')
end = 10000000
findSw = False

with open('dupchk.hash') as data:
   lines = data.readlines()

for i in range(end):
    line = fsrc.readline()
    #print(str(i) + ':' + line)
    for j in range(10000000):
        #print(line + lines[j])
        if (line == lines[j] and i != j):
            print(str(i) + ':' + line)
            findSw = True
            break

    print(i)
    if findSw:
        break

data.close()    
fsrc.close()



