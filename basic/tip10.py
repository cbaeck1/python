
# tip 1
condition = True
# if condition:
#   x = 1
# else:
#   x = 0
# print(x)

x = 1 if condition else 0
print(x)

# tip 2 
num1 = 10_000_000_000
num2 = 100_000_000
total = num1 + num2

print(total)
print(f'{total:,}')

# tip 3
# f = open('test.txt','r')
# f.close()

with open('test.txt','r') as f:
  file_contents = f.read()
 
words = file_contents.split(' ')
word_count = len(words)
print(word_count)

# itertools
# tip 4
names = ['Min', 'Soo', 'Baeck', 'Ho']
# index = 0
# for name in names:
#  print(index, name)
#  index += 1

for index, name in enumerate(names, start=0):
  print(index, name)

names = ['Peter Parker', 'Clark Kent', 'Wade Wilson', 'Bruce Wayne']
heroes = ['Spiderman', 'Superman', 'Deadpool', 'Batman']
#for index, name in enumerate(names):
#  hero = heroes[index]
#  print(f'{name} is actually {hero}')

for name, hero in zip(names, heroes):
  print(f'{name} is actually {hero}')

universes = ['Marvel', 'DC', 'Marvel', 'DC']
for name, hero, universe in zip(names, heroes, universes):
  print(f'{name} is actually {hero} from {universe}')

for value in zip(names, heroes, universes):
  print(value)

# Normal
# tip 5
items = (1, 2)
print(items)

a, b = (1, 2)
print(a, b)

# unpacking
a, b = (1, 2)
print(a)

a, _ = (1, 2)
print(a)

# not enough values to unpack (expected 3, got 2)
# a, b, c = (1, 2)
# too many values to unpack (expected 3)
# a, b, c = (1, 2, 3, 4, 5)

a, b, *c = (1, 2, 3, 4, 5)
print(a, b, c)

a, b, *_ = (1, 2, 3, 4, 5)
print(a, b)

a, b, *c, d = (1, 2, 3, 4, 5)
print(a, b, c, d)

a, b, *_, d = (1, 2, 3, 4, 5)
print(a, b, d)

# tip 6
class Person():
  pass

person = Person()
# person.first = 'Soo'
# person.last = 'Min'
# print(person.first, person.last)

first_key = 'first'
first_val = 'Soo'

setattr(person, 'first', 'Ho')
print(person.first)

setattr(person, first_key, first_val)
print(person.first)
first = getattr(person, first_key)
print(first)

person_info = {'first': 'Ho', 'last': 'Baeck'}
for key, value in person_info.items():
  setattr(person, key, value)
print(person.first, person.last)

for key in person_info.keys():
  print(getattr(person, key))







