import hashlib

line = 'ncc최은숙7101132'

encoded_string = line.encode()
hexdigest = hashlib.sha256(encoded_string).hexdigest().upper()

print(hexdigest)