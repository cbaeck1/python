# 
# f = open(sys.argv[1], 'r') 

f = open('data_fwf.txt', 'r', encoding='UTF8') 
lines = f.readlines()
for line in lines:
    name = line[0:2]
    age = line[2:5]
    city = line[5:8]
    print(name+age+city)


f = open('data_fwf2.txt', 'r') 
lines = f.readlines()
for line in lines:
    line = line.rstrip('\n')
    item = line.split(" ")
    print(item)
    user_name = item[0]
    user_age = item[1]
    user_city = item[2]
    user_sex = item[3]
    print(user_name+','+user_age+','+user_city+','+user_sex)

    