import hashlib

# E3CD087091E9404A9DB075FF43CEB1A2전윤철3809055
# AB04382BD6B54D21023D3D99A2B9E389C5E375AB7312AB881104E46137BC8BB0 
# AB04382BD6B54D21023D3D99A2B9E389C5E375AB7312AB881104E46137BC8BB0

string = 'E3CD087091E9404A9DB075FF43CEB1A2전윤철3809055'
print(string)
encoded_string = string.encode()
print(encoded_string)
hexdigest = hashlib.sha256(encoded_string).hexdigest()
print(hexdigest.upper())
 
