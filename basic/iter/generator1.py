# 제너레이터

# 제너레이터는 반복자(iterator)와 같은 루프의 작용을 컨트롤하기 위해 쓰여지는 특별한 함수 또는 루틴이다. 
# 사실 모든 제너레이터는 반복자이다. 제너레이터는 배열이나 리스트를 리턴하는 함수와 비슷하며, 
# 호출을 할 수 있는 파라메터를 가지고 있고, 연속적인 값들을 만들어 낸다. 
# 하지만 한번에 모든 값을 포함한 배열을 만들어서 리턴하는 대신에 yield 구문을 이용해 
# 한 번 호출될 때마다 하나의 값만을 리턴하고, 이런 이유로 일반 반복자에 비해 아주 작은 메모리를 필요로 한다. 
# 간단히 얘기하면 제너레이터는 반복자와 같은 역할을 하는 함수이다
# by 위키피티아

# 일반함수가 호출되면 코드의 첫 번째행 부터 시작하여 리턴(return) 구문이나, 예외(exception) 
# 또는 (리턴을 하지않는 함수이면) 마지막 구문을 만날때까지 실행된 후, 호출자(caller)에게 모든 컨트롤을 리턴합니다. 
# 그리고 함수가 가지고 있던 모든 내부 함수나 모든 로컬 변수는 메모리상에서 사라집니다. 
# 같은 함수가 다시 호출되면 모든 것은 처음부터 다시 새롭게 시작됩니다

# 하나의 일을 마치면 자기가 했던 일을 기억하면서 대기하고 있다가 다시 호출되면 전의 일을 계속 이어서 하는 똑똑한 함수

# 일반함수
def square_common(nums):
    result = []
    for i in nums:
        result.append(i * i)
    return result

my_nums = square_common([1, 2, 3, 4, 5])
# [1, 4, 9, 16, 25]
print(my_nums) 

# 제너레이터 함수
def square_yield(nums):
    for i in nums:
        yield i * i

my_nums = square_yield([1, 2, 3, 4, 5])
# <generator object square_yield at 0x000001D03B9B76D0>
print(my_nums) 
for num in my_nums:
  print(num) 

# list comprehension
my_nums = [x * x for x in [1, 2, 3, 4, 5]]
# [1, 4, 9, 16, 25]
print(my_nums) 
for num in my_nums:
  print(num) 

# 제너레이터
my_nums = (x * x for x in [1, 2, 3, 4, 5])
# [1, 4, 9, 16, 25]
print(my_nums) 
print(list(my_nums))
# * 주의 : 제너레이트는 한번 사용하면 다시 사용할 수 없다.
# my_nums에 아무것도 없어 출력되지 않음
for num in my_nums:
  print(num) 



