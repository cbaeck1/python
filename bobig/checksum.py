import hashlib

def calc_file_hash(path):
    f = open(path, 'rb')
    data = f.read()
    hash = hashlib.md5(data).hexdigest()
    return hash

# if __name__ == "__main__":
hash_val_a = calc_file_hash("C:/data/test_data_a.txt")
print(hash_val_a)

hash_val_b = calc_file_hash("C:/data/test_data_b.txt")
print(hash_val_b)

