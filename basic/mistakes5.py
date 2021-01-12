
# 1. Tab, space : unindent does not match any outer indentation level
nums = [11, 30, 44, 54]

for num in nums:
#    square = num ** 2
  square = num ** 2
  print(square)

# 2. 라이블러리와 같은 이름의 파일, 함수와 같은 이름의 변수
# math.py
# Exception has occurred: TypeError 'float' object is not callable
from math import radians, sin

rads = radians(90)
print(sin(rads))

# radians = radians(90)
rad45 = radians(45)
print(rad45)

# Exception has occurred: AttributeError
# partially initialized module 'logging' has no attribute 'getLogger' (most likely due to a circular import)

# 3. default argument
# def add_employee(emp, emp_list=[]):
#   emp_list.append(emp)
#   print(emp_list)

def add_employee(emp, emp_list=None):
  if emp_list is None:
    emp_list = []
  emp_list.append(emp)
  print(emp_list)

emps = ['John', 'Jane']
# add_employee('Baeck', emps)
# add_employee('John', emps)

# ['Baeck'] 
# ['Baeck', 'John'] 
# ['Baeck', 'John', 'Jane']
add_employee('Baeck')
add_employee('John')
add_employee('Jane')

# August 08, 2020 10:32:29
# August 08, 2020 10:32:29
# August 08, 2020 10:32:29
import time
from datetime import datetime
def display_time_same(time=datetime.now()):
  print(time.strftime('%B %d, %Y %H:%M:%S'))

display_time_same()
time.sleep(1)
display_time_same()
time.sleep(1)
display_time_same()

def display_time(time=None):
  if time is None:
    time=datetime.now()
  print(time.strftime('%B %d, %Y %H:%M:%S'))

display_time()
time.sleep(1)
display_time()
time.sleep(1)
display_time()

# 4.  
# <zip object at 0x00000230C323A300>
names = ['Peter Parker', 'Clark Kent', 'Wade Wilson', 'Bruce Wayne']
heroes = ['Spiderman', 'Superman', 'Deadpool', 'Batman']

identites = zip(names, heroes)
print(identites)
print(type(identites))
# print(list(identites))

identites = list(zip(names, heroes))
print(identites)

for identity in identites:
  print('{} is actually {}!'.format(identity[0], identity[1]))

# 5.
# import

from html import *
from glob import *

print(help(escape))

