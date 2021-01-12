#-*- coding: utf-8 -*-

import datetime
import time

# 로그파일 한글 문제 
# logging.basicConfig(filename='{}.log'.format(original_function.__name__), level=logging.INFO)

def my_logger(original_function):
    import logging
    logging.basicConfig(handlers=[logging.FileHandler(filename='{}.log'.format(original_function.__name__), encoding='utf-8')], 
        level=logging.INFO)
    
    def wrapper(*args, **kwargs):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        logging.info('[{}] 실행결과 args - {}, kwargs - {}'.format(timestamp, args, kwargs))
        return original_function(*args, **kwargs)
    return wrapper

@my_logger
def display_info(name, age):
    time.sleep(1)
    print('display_info({}, {}) 함수가 실행됐습니다.'.format(name, age))

display_info('John', 25)


