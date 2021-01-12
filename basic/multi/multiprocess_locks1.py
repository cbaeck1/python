from multiprocessing import Process, Lock, Value
import time

def add_500_no_lock(total):
  for i in range(100):
    time.sleep(0.01)
    total.value += 5

def sub_500_no_lock(total):
  for i in range(100):
    time.sleep(0.01)
    total.value -= 5

def add_500_lock(total, lock):
  for i in range(100):
    time.sleep(0.01)
    lock.acquire()
    total.value += 5
    lock.release()

def sub_500_lock(total, lock):
  for i in range(100):
    time.sleep(0.01)
    lock.acquire()
    total.value -= 5
    lock.release()

# 파이썬은 main문이 없는 대신에, 들여쓰기가 되지 않은 Level0의 코드를 가장 먼저 실행
# __name__ : 현재 모듈의 이름을 담고있는 내장 변수
#   직접 실행된 모듈인 경우 __main__  이라는 값을
#   import된 모듈은 파일명 값을 가진다.
if __name__ == '__main__':
  # no lock
  total = Value('i', 500)
  print(total.value)
  
  start = time.perf_counter()
  add_proces = Process(target=add_500_no_lock, args=(total,))
  sub_proces = Process(target=sub_500_no_lock, args=(total,))

  add_proces.start()
  sub_proces.start()

  add_proces.join()
  sub_proces.join()
  print(total.value)

  finish = time.perf_counter()
  print(f'Finished in {round(finish-start, 2)} second(s)')


  # lock
  total = Value('i', 500)
  print(total.value)
  
  start = time.perf_counter()
  lock = Lock()
  add_proces = Process(target=add_500_lock, args=(total, lock))
  sub_proces = Process(target=sub_500_lock, args=(total, lock))

  add_proces.start()
  sub_proces.start()

  add_proces.join()
  sub_proces.join()
  print(total.value)

  finish = time.perf_counter()
  print(f'Finished in {round(finish-start, 2)} second(s)')


