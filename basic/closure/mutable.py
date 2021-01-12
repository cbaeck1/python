'''
mutable은 값이 변한다는 뜻이고, immutable은 값이 변하지 않는다는 의미이다. 자료형마다 특징이
- 숫자형 (Number) : immutable
- 문자열 (String) : immutable
- 리스트 (List) : mutable
- 튜플 (Tuple) : immutable
- 딕셔너리 (Dictionary) : mutable
숫자, 문자열, 튜플은 변경이 불가능하고 리스트와 딕셔너리는 변경이 가능하다. 
위 코드의 $y=x$부분에서 y와 x가 같은 주소 값을 가리키게 되는데, 
리스트의 [:]나 deepcopy 함수를 이용하면 같은 객체를 공유하지 않도록 사용 가능하다.

'''

# 숫자
print('숫자=================')
x = 1
print('Address of x is: {}'.format(id(x)))
y = x
print('Address of y is: {}'.format(id(y)))
y += 3
# 값을 변경하면 새로운 메모리를 할당하여 저장
print('Address of y is: {}'.format(id(y)))

# 문자
print('문자=================')
a = 'abc'
print(a)
print('Address of a is: {}'.format(id(a))) 
# a 의 첫번째 위치의 값을 변경할 수 없다 : 숫자형은 immutable 하기 때문
# Exception has occurred: TypeError 'str' object does not support item assignment
# a[0] = 'x'

# 값을 변경하면 (새로운 값을 할당하면) 새로운 메모리를 할당하여 저장
a = 'xyz'
print(a)
print('Address of a is: {}'.format(id(a))) 

# 리스트
print('리스트=================')
x = [1,2]  
y = x     
z = x[:]  # x와 z가 같은 객체를 공유하지 않는다.(리스트만 가능)
# deepcopy를 이용하면 x와 cpy가 같은 객체를 공유하지 않는다.
import copy
dcpy = copy.deepcopy(x)

print('Address of x is: {}'.format(id(x)))
print('Address of y is: {}'.format(id(y)))
print('Address of z is: {}'.format(id(z)))
print('Address of dcpy is: {}'.format(id(dcpy)))
x += [3]
# 값을 추가해도 같은 메모리
print('Address of x is: {}'.format(id(x)))

# output 에 값을 추가할 때마다 새로운 메모리를 할당
# string 과 stringBuffer 의 성능차이
employees = ['Corey', 'John', 'Rick', 'Steve', 'Carl', 'Adam']
output = '<ul>\n'

for employee in employees:
    output += '\t<li>{}</li>\n'.format(employee)
    print('Address of output is {}'.format(id(output)))

output += '</ul>'

print(output)
print('\n')

a = [1,2,3,4,5]
print(a)
print('Address of a is: {}'.format(id(a)))

a[0] = 6
print(a)
print('Address of a is: {}'.format(id(a)))

