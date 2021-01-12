# -*- coding: utf-8 -*-

# 6. 매직 메소드(Magic Method)

class Dog(object):
    def __init__(self, name, age):
        print('이름: {}, 나이: {}'.format(name, age))
        
dog_1 = Dog('Pink', '12')