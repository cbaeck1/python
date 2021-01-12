import logging
import datetime

logging.basicConfig(filename='log/example.log', level=logging.INFO)

def logger(func):
  def log_func(*args):
    logging.info('{} : Running "{}" with arguments {}'.format(datetime.datetime.now(), func.__name__, args))
    print('Running "{}" with arguments {} : '.format(func.__name__, args), func(*args))
    # return func(*args)
  return log_func

# 함수를 정의
def add(x, y):
  return x+y

def sub(x, y):
  return x-y

# 1. log 기록하지 않고 실행
a = add(3, 3)
b = sub(10, 5)
print(a, b)

# 2. log를 파일로 기록하도록 함수를 생성 : closure 적용
add_wlogger = logger(add)
sub_wlogger = logger(sub)

# log 기록하는 함수를 실행
# logger 함수의 inner function 인 log_func 에 return 이 없으면 a1 은 None이고 
#                                            return 을 func(*args) 으로 정의하면 a1 은 함수의 결과값이된다
a1 = add_wlogger(3, 3)
a2 = add_wlogger(4, 5)
b1 = sub_wlogger(10, 5)
b2 = sub_wlogger(20, 10)
print(a1, a2, b1, b2)

# 또 다른 방식
def add_logger(x, y):
  return x+y

def sub_logger(x, y):
  return x-y

add = logger(add_logger)
sub = logger(sub_logger)

a1 = add(1, 1)
a2 = add(2, 3)
b1 = sub(100, 50)
b2 = sub(200, 100)
print(a1, a2, b1, b2)

