# 데코레이터
# 자신의 방을 예쁜 벽지나 커튼으로 장식을 하듯이, 기존의 코드에 여러가지 기능을 추가하는 방식

# 클로저
def outer_function(msg):
    def inner_function():
        print(msg)
    return inner_function

# 함수를 retrun 
print(outer_function('Bye').__name__)
# print 가 안되는
outer_function('Bye')
# print 가 되는
outer_function('Bye')()

hi_func = outer_function('Hi')
bye_func = outer_function('Bye')
print(hi_func.__name__)

# print 가 되는
hi_func()
bye_func()

# 
def decorator_function(original_function):  #1
    def wrapper_function():  #5
        return original_function()  #7
    return wrapper_function  #6

def display():  #2
    print('display 함수가 실행됐습니다.')  #8

# 함수를 직접 실행
display()
decorated_display = decorator_function(display)  #3
# 함수를 데코레이션 함수를 통해 실행
# 이미 만들어져 있는 기존의 코드를 수정하지 않고도, 래퍼(wrapper) 함수를 이용하여 여러가지 기능을 추가할 수가 있기 때문입니다
decorated_display()  #4

# 
def decorator_func(original_function):
    def wrapper_function():
        print('{} 함수가 호출되기전 입니다.'.format(original_function.__name__))
        return original_function()
    return wrapper_function

def display_1():
    print('display_1 함수가 실행됐습니다.')

def display_2():
    print('display_2 함수가 실행됐습니다.')

display_1 = decorator_func(display_1)  #1
display_2 = decorator_func(display_2)  #2

display_1()
print('--------------------')
display_2()

# "@" 심볼과 데코레이터 함수의 이름으로 구현
def decorator_func2(original_function):
    def wrapper_function():
        print('{} 함수가 호출되기전 입니다.'.format(original_function.__name__))
        return original_function()
    return wrapper_function

@decorator_func2  #1
def display_1():
    print('display_1 함수가 실행됐습니다.')

@decorator_func2  #2
def display_2():
    print('display_2 함수가 실행됐습니다.')

# display_1 = decorator_function(display_1)  #3
# display_2 = decorator_function(display_2)  #4

display_1()
print('--------------------')
display_2()


# argument가 있는 함수를 데코레이션하는 방식 : 인수를 추가 *args, **kwargs
def decorator_func3(original_function):
    def wrapper_function(*args, **kwargs):
        print('{} 함수가 호출되기전 입니다.'.format(original_function.__name__))
        return original_function(*args, **kwargs)
    return wrapper_function

@decorator_func3
def display():
    print('display_1 함수가 실행됐습니다.')

@decorator_func3
def display_info(name, age):
    print('display_info({}, {}) 함수가 실행됐습니다.'.format(name, age))

display()
print('--------------------')
display_info('John', 25)

# 클래스 형식으로 데코레이션 하기

class DecoratorClass:  #1
    def __init__(self, original_function):
        self.original_function = original_function

    def __call__(self, *args, **kwargs):
        print('{} 함수가 호출되기전 입니다.'.format(self.original_function.__name__))
        return self.original_function(*args, **kwargs)

@DecoratorClass  #2
def display():
    print('display_1 함수가 실행됐습니다.')

@DecoratorClass  #3
def display_info(name, age, sex):
    print('display_info({}, {}, {}) 함수가 실행됐습니다.'.format(name, age, sex))

display()
print('--------------------')
display_info('John', 25, 'male')


