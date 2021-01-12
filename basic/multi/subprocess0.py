import subprocess

# windows : shell=True
subprocess.run('dir', shell=True)

# argument
subprocess.run(['dir', '*'], shell=True)

# stdout
p1 = subprocess.run(['dir', '*'], shell=True)
print(p1)
print(p1.args)
print(p1.returncode)
print(p1.stdout, p1.stderr)

#
p1 = subprocess.run(['dir', '*'], shell=True, capture_output=True)
# byte 
print(p1.stdout)
# 'utf-8' codec can't decode byte 0xb5 in position 3: invalid start byte
# print(p1.stdout.decode())

p1 = subprocess.run(['dir', '*'], shell=True, capture_output=True, text=True)
print(p1.stdout)

p1 = subprocess.run(['dir', '*'], shell=True, stdout=subprocess.PIPE, text=True)
print(p1.stdout)

with open('output.txt', 'w', encoding='utf-8') as f:
  p1 = subprocess.run(['dir', '*'], shell=True, stdout=f, text=True)

# 
p1 = subprocess.run(['dir', '*', 'dd'], shell=True, capture_output=True, text=True)
print(p1.stderr)

p1 = subprocess.run(['dir', '*', 'dd'], shell=True, capture_output=True, text=True, check=True)
print(p1.stderr)

p1 = subprocess.run(['dir', '*', 'dd'], shell=True, stderr=subprocess.DEVNULL)
print(p1.stderr)