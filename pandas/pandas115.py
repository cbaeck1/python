# Python 중고급 - map, filter, reduce 
# 파이썬의 기초를 익힌 후, 파이썬의 중고급 문법을 선택적으로 배운다면 기본 문법으로도 구현할 수 있는 로직들을 더욱 쉽고, 간편하게 구현해볼 수 있습니다. 이번 포스팅에서는 list를 다루는 함수인 map, filter, reduce에 대해 간단하게 정리해보겠습니다. 물론 map, filter, reduce를 안 쓰고 코딩하는 것도 가능하며, 그것이 편하시면 그렇게 하셔도 좋습니다. 하지만 이를 사용하는 프로그래머나 데이터 분석가들이 꽤 있기 때문에 이러한 문법들을 알아두면 기존의 코드를 이해하는데 큰 도움이 될 수 있을 것입니다. 

# Map : map의 경우, list의 element에 함수를 적용시켜 결과를 반환하고 싶을 때 사용합니다. 만약, 어떤 리스트의 원소를 제곱한 새로운 리스트를 만들고 싶다면, 일반적인 C언어 스타일의 해결법은 아래와 같습니다. 
# Map function
# 문제와 일반적인 해결법
import numpy as np
import pandas as pd

items = [1, 2, 3, 4, 5]
squared = []
for i in items:
    squared.append(i**2)
# 하지만 map function을 통해 짧게 구현할 수 있습니다. map함수의 구조는 map(function_to_apply, list_of_inputs) 입니다.

items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
# 위 코드는 items 리스트의 element 각각에 대해 제곱을 해서 새로운 리스트를 만들어 반환합니다. filter와 reduce도 이와 같은 구조를 갖습니다. 
# 이 때, map 앞에 list 함수를 통해 list 자료형으로 바꾸는 이유는 map 이 반환하는 것이 실제로는 list 자료형이 아니기 때문입니다. map 함수는 Iterator 를 반환하는데, 이를 list로 변환해서 list 자료형으로 만드는 것입니다. Iterator는 next() 함수를 갖는 파이썬 객체로 꼭 메모리에 올릴 데이터만 올려서 메모리를 효율적으로 이용할 수 있는 파이썬의 대표적인 객체입니다. 바로 list로 반환하는 것이 아니라 Iterator로 보다 상위의 객체를 리턴하는 것은, 다른 map 함수의 리턴값을 리스트가 아닌 다른 자료구조로 변환시킬 수도 있도록 하기 위해서입니다. 예를 들어, set 자료구조로도 변환시킬 수 있습니다. 

map_it = map(lambda x: x**2, items)
next(map_it)
# map 함수의 결과는 Iterator 이므로, next 함수를 위와 같이 실행할 수 있습니다. 위 코드의 결과는 1입니다.
# Iterator 에 대해서는 다음에 다루어 보도록 하겠습니다. 

# Filter : Filter의 경우, list의 element에서 어떤 함수의 조건에 일치하는 값만 반환하고 싶을 때 사용합니다. 
# As the name suggests, filter creates a list of elements for which a function returns true.
# 만약 어떤 리스트에서 '음수만 골라내고 싶다' 라고 할 때, filter 함수를 사용한 코딩 방법은 아래와 같습니다. 
number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero)
# 이 때, less_than_zero 는 -5, -4, -3, -2, -1 을 갖습니다. 
# 만약 Map과 Filter를 보고 저거 list comprehension 으로 할 수 있는 거 아니야? 라고 생각하실 수 있습니다. 맞습니다. Map과 Filter가 자신과 맞지 않다고 생각하는 경우 list comprehension 만으로도 위 코드들을 훨씬 더 간결하게 구현할 수 있습니다.

# Map을 list comprehension으로 구현
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
print(squared)
 
squared = [x**2 for x in items]
print(squared)
 
# Filter를 list comprehension으로 구현
number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero)
 
less_than_zero = [x for x in number_list if x <0]
print(less_than_zero)
 
# 위 코드처럼 list comprehension 만을 통해 map, filter가 하는 것을 할 수 있습니다. 
# Filter가 마음에 들지 않는 경우, list comprehension을 쓸 수도 있습니다 하지만, map, filter가 있다는 것을 알아 두면 좋습니다.

# Reduce : reduce는 어떤 list에 대해서 computation을 해서 결과를 내보낼 때, 즉 결과를 어떤 함수로 computation해서 결과를 축약하기 위해서 사용됩니다. 이 때, reduce함수에 input으로 들어가는 함수는 두 element를 연산하는 로직을 넣어야 합니다. 이것은 list comprehension으로 하지 못하는 것입니다. 
# 아래 코드는 reduce함수를 이용하는 것을 포함하여 파이썬으로 1-100 까지의 합을 구하는 총 3가지 방법입니다. 
# 1-100 까지의 합을 구하라 
# C 언어 스타일의 해법
sum_value = 0
for i in range(1, 101) :
    sum_value += i 
print(sum_value)
 
# python 스러운 해법
from functools import reduce
sum_value = reduce((lambda x,y : x+y), [x for x in range(1,101)])
print(sum_value)
 
# 하지만 Data scientist가 가장 먼저 생각해는 해법 
print(np.sum([x for x in range(1,101)]))


# 참고 : Intermediate python (http://book.pythontips.com/en/latest/)


