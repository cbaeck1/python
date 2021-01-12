import subprocess

# windows : shell=True
p1 = subprocess.run(['type', 'test.txt'], shell=True, capture_output=True, text=True)
print(p1.stdout)

p2 = subprocess.run(['find', 'five'], shell=True, capture_output=True, text=True, input=p1.stdout)
print(p2.stdout, p2.stderr)
