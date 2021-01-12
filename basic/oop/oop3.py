# -*- coding: utf-8 -*-

# 3. 클래스 변수(Class Variable)

class Employee(object):

    raise_amount = 1.1  #1 클래스 변수 정의

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first.lower() + '.' + last.lower() + '@schoolofweb.net'
        
    def full_name(self):
        return '{} {}'.format(self.first, self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay * 1.1)  #1 연봉을 10% 인상합니다.

emp_1 = Employee('Sanghee', 'Lee', 50000)
emp_2 = Employee('Minjung', 'Kim', 60000)

print(emp_1.pay)  # 기존 연봉
emp_1.apply_raise()  # 인상률 적용
print(emp_1.pay)  # 오른 연봉



