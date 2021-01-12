import hashlib, re

pattern = re.compile(r'\s+')

# E3CD087091E9404A9DB075FF43CEB1A2전윤철3809055
# AB04382BD6B54D21023D3D99A2B9E389C5E375AB7312AB881104E46137BC8BB0 
# AB04382BD6B54D21023D3D99A2B9E389C5E375AB7312AB881104E46137BC8BB0

key = 'E3CD087091E9404A9DB075FF43CEB1A2'
case1 = '홍길 9912311'
case2 = '홍 길9912311'
case3 = ' 홍길9912311'
case4 = '홍길  9912311'
case5 = '홍  길9912311'
case6 = '  홍길9912311'
case7 = '홍길	9912311'
case8 = '홍	길9912311'
case9 = '	홍길9912311'
case10 = '홍길9912311'

#print(key+case10)
#encoded_string = (key+case10).encode()
#print(encoded_string)
hexdigest1 = hashlib.sha256((key+case1).encode()).hexdigest()
hexdigest2 = hashlib.sha256((key+case2).encode()).hexdigest()
hexdigest3 = hashlib.sha256((key+case3).encode()).hexdigest()
hexdigest4 = hashlib.sha256((key+case4).encode()).hexdigest()
hexdigest5 = hashlib.sha256((key+case5).encode()).hexdigest()
hexdigest6 = hashlib.sha256((key+case6).encode()).hexdigest()
hexdigest7 = hashlib.sha256((key+case7).encode()).hexdigest()
hexdigest8 = hashlib.sha256((key+case8).encode()).hexdigest()
hexdigest9 = hashlib.sha256((key+case9).encode()).hexdigest()
hexdigest10 = hashlib.sha256((key+case10).encode()).hexdigest()
print(hexdigest1.upper())
print(hexdigest2.upper())
print(hexdigest3.upper())
print(hexdigest4.upper())
print(hexdigest5.upper())
print(hexdigest6.upper())
print(hexdigest7.upper())
print(hexdigest8.upper())
print(hexdigest9.upper())
print(hexdigest10.upper())


hexdigest1 = hashlib.sha256(''.join((key+case1).split()).encode()).hexdigest()
hexdigest2 = hashlib.sha256(''.join((key+case2).split()).encode()).hexdigest()
hexdigest3 = hashlib.sha256(''.join((key+case3).split()).encode()).hexdigest()
hexdigest4 = hashlib.sha256(''.join((key+case4).split()).encode()).hexdigest()
hexdigest5 = hashlib.sha256(''.join((key+case5).split()).encode()).hexdigest()
hexdigest6 = hashlib.sha256(''.join((key+case6).split()).encode()).hexdigest()
hexdigest7 = hashlib.sha256(''.join((key+case7).split()).encode()).hexdigest()
hexdigest8 = hashlib.sha256(''.join((key+case8).split()).encode()).hexdigest()
hexdigest9 = hashlib.sha256(''.join((key+case9).split()).encode()).hexdigest()

print(hexdigest1.upper())
print(hexdigest2.upper())
print(hexdigest3.upper())
print(hexdigest4.upper())
print(hexdigest5.upper())
print(hexdigest6.upper())
print(hexdigest7.upper())
print(hexdigest8.upper())
print(hexdigest9.upper())


 
