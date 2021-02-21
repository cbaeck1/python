path = 'data/COVID-19.csv'
with open(path, 'rb') as f:
    buf  = f.read()
    print(buf)


with open(path, 'r', encoding='euc-kr') as f:
    lines  = f.read()