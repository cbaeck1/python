# -*- coding: utf-8 -*-

# 2. 클래스와 인스턴스(Class and Instance)
# 오브젝트란 속성과 같은 여러가지의 데이터와 함수(오브젝트 안에서는 메소드라고 부릅니다.)를 포함한 하나의 데이터 구조로
# 데이터를 조금 더 쉽게 다루기 위해서 "네임스페이스"라는 것을 이용하여 만든 논리적인 집합이다.
# 파이썬에서 오브젝트들은 변수에 할당될 수도 있고, 함수의 인자로 전달될 수도 있는 퍼스트 클래스 오브젝트이다

# 사전을 사용하는 경우
student = {'name': '민수정', 'year': 2, 'class': 3, 'student_id': 35}
print('{}, {}학년 {}반 {}번'.format(student['name'], student['year'], student['class'], student['student_id']))

# 클래스를 사용하는 경우
class Student(object):
    def __init__(self, name, year, class_num, student_id):  # 파이썬 키워드인 class는 인수 이름으로 사용하지 못 합니다.
        self.name = name
        self.year = year
        self.class_num = class_num
        self.student_id = student_id
        
    def introduce_myself(self):
        return '{}, {}학년 {}반 {}번'.format(self.name, self.year, self.class_num, self.student_id)
    
student = Student('민수정', 2, 3, 35)
print(student.introduce_myself())

# 모듈을 사용하는 경우
import student
print('{}, {}학년 {}반 {}번'.format(student.name, student.year, student.class_id, student.student_id))


# dir()는 파이썬의 표준 내장 함수입니다. 
# 이 함수는 인자가 없을 경우에는 모듈 레벨의 지역변수를, 
# 인자가 있을 경우에는 인자(오브젝트)의 모든 속성과 메소드를 보여줍니다. 
# 이 함수는 디버깅을 할 때 아주 많이 쓰이는 중요한 함수입니다.
text = 'string'
print(dir(text))

# text의 클래스 확인
print(text.__class__)  # <type 'str'>
# text가 str의 인스턴스 오브젝트인지 확인
print(isinstance(text, str))  # True
# 메소드 확인
print(text.upper())  # STRING


def my_function():
    '''my_function에 대한 설명입니다~!'''
    pass

# my_function의 속성 확인
print(dir(my_function), '\n')
# my_function의 docstring 출력
print(my_function.__doc__, '\n')

# my_function에 새로운 속성 추가
my_function.new_variable = '새로운 변수입니다.'

# 추가된 속성 확인
print(dir(my_function), '\n')
# 추가한 속성값 출력
print(my_function.new_variable, '\n')

# 
class Employee(object):
    pass

emp_1 = Employee()
emp_2 = Employee()

# emp_1과 emp_2는 다른 메모리 주소값을 가진 별개의 오브젝트입니다.
print(id(emp_1))
print(id(emp_2))

# emp_1과 emp_2는 같은 클래스의 인스턴스인 것을 확인합니다.
class_of_emp_1 = emp_1.__class__
class_of_emp_2 = emp_2.__class__
print(id(class_of_emp_1))
print(id(class_of_emp_2))


