
# sha256 해시함수 사용 복호화

import hashlib


'''
print(hashlib.sha256('A'.encode()).hexdigest().upper())

str = "AB04382BD6B54D21023D3D99A2B9E389C5E375AB7312AB881104E46137BC8BB0"
for i in range(64):
 str = str.replace(hashlib.sha256(chr(i).encode()).hexdigest(), chr(i))
print(str)
#  Result
#  E3CD087091E9404A9DB075FF43CEB1A2전윤철3809055

string = 'a'
#print(string)
encoded_string = string.encode()
#print(encoded_string)
hexdigest = hashlib.sha256(encoded_string).hexdigest()
print(hexdigest.upper())
#  Result
#  F45DE51CDEF30991551E41E882DD7B5404799648A0A00753F44FC966E6153FC1
'''

string = 'a'
#print(string)
encoded_string = string.encode()
#print(encoded_string)
hexdigest = hashlib.sha256(encoded_string).hexdigest()
print(hexdigest)

str = 'ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6'
for i in range(123):
 #print(chr(i))
 #print(hashlib.sha256(chr(i).encode()).hexdigest().upper())
 str = str.replace(hashlib.sha256(chr(i).encode()).hexdigest(), chr(i))
 #print(str)

print(str)
#  Result
#  ac

