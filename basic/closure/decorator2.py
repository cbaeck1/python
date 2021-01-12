# -*- coding: utf-8 -*-

import datetime
import time

# 로그파일을 기록하지 않음, 화면 출력은 됨 <-- my_timer_wrong1 만 실행
def my_logger_wrong1(original_function):
    import logging
    logging.basicConfig(handlers=[logging.FileHandler(filename='{}.log'.format(original_function.__name__), encoding='utf-8')], 
        level=logging.INFO)
    
    def wrapper_logger(*args, **kwargs):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        logging.info('[{}] 실행결과 args - {}, kwargs - {}'.format(timestamp, args, kwargs))
        return original_function(*args, **kwargs)
    return wrapper_logger

def my_timer_wrong1(original_function):
    import time
    
    def wrapper_timer(*args, **kwargs):
        t1 = time.time()
        result = original_function(*args, **kwargs)
        t2 = time.time() - t1
        print('{} 함수가 실행된 총 시간: {} 초'.format(original_function.__name__, t2))
        return result
    return wrapper_timer

@my_logger_wrong1
@my_timer_wrong1  
def display_info(name, age):
    time.sleep(1)
    print('display_info({}, {}) 함수가 실행됐습니다.'.format(name, age))

display_info('John', 25)
display_info('Jimmy', 30)
print('------------------------------------------')

# 로그파일을 기록, 화면 출력도 됨
# 사용하는 함수 wrapper_logger 
def my_logger_wrong2(original_function):
    import logging
    logging.basicConfig(handlers=[logging.FileHandler(filename='{}.log'.format(original_function.__name__), encoding='utf-8')], 
        level=logging.INFO)
    
    def wrapper_logger(*args, **kwargs):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        logging.info('[{}] 실행결과 args - {}, kwargs - {}'.format(timestamp, args, kwargs))
        return original_function(*args, **kwargs)
    return wrapper_logger

def my_timer_wrong2(original_function):
    import time
    
    def wrapper_timer(*args, **kwargs):
        t1 = time.time()
        result = original_function(*args, **kwargs)
        t2 = time.time() - t1
        print('{} 함수가 실행된 총 시간: {} 초'.format(original_function.__name__, t2))
        return result
    return wrapper_timer

@my_timer_wrong2  
@my_logger_wrong2
def display_info(name, age):
    time.sleep(1)
    print('display_info({}, {}) 함수가 실행됐습니다.'.format(name, age))

display_info('John', 25)
display_info('Jimmy', 30)
print('------------------------------------------')


# 복수의 데코레이터를 스택해서 사용하면 아래쪽 데코레이터부터 실행되는데, 
# my_logger 가 먼저 실행되고 리턴 wrapper_logger 가 my_timer 에게 인자로 전달하기 때문에 생기는 현상입니다.
# original_function 은 물론 wrapper_logger 함수와 같습니다.
# 위와 같은 현상을 방지하기 위해서 functools 모듈의 wraps 데코레이터를 사용해야 함

from functools import wraps

def my_logger(original_function):
    import logging
    logging.basicConfig(handlers=[logging.FileHandler(filename='{}.log'.format(original_function.__name__), encoding='utf-8')], 
        level=logging.INFO)

    @wraps(original_function)
    def wrapper_logger(*args, **kwargs):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        logging.info('[{}] 실행결과 args - {}, kwargs - {}'.format(timestamp, args, kwargs))
        return original_function(*args, **kwargs)
    return wrapper_logger

def my_timer(original_f):
    import time
    
    @wraps(original_f)
    def wrapper_timer(*args, **kwargs):
        t1 = time.time()
        result = original_f(*args, **kwargs)
        t2 = time.time() - t1
        print('{} 함수가 실행된 총 시간: {} 초'.format(original_f.__name__, t2))
        return result
    return wrapper_timer

@my_timer  
@my_logger
def display_info(name, age, sex='male'):
    time.sleep(1)
    print('display_info({}, {}, {}) 함수가 실행됐습니다.'.format(name, age, sex))

display_info('John', 25)
display_info('Sara', 30, 'female')


