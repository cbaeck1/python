import multiprocessing
import time

def do_something():
  print('Sleeping 1 second...')
  time.sleep(1)
  print('Done Sleeping...')

# 파이썬은 main문이 없는 대신에, 들여쓰기가 되지 않은 Level0의 코드를 가장 먼저 실행
# __name__ : 현재 모듈의 이름을 담고있는 내장 변수
#   직접 실행된 모듈인 경우 __main__  이라는 값을
#   import된 모듈은 파일명 값을 가진다.
if __name__ == '__main__':
  start = time.perf_counter()
  p1 = multiprocessing.Process(target=do_something)
  p2 = multiprocessing.Process(target=do_something)

  p1.start()
  p2.start()

  p1.join()
  p2.join()

  finish = time.perf_counter()
  print(f'Finished in {round(finish-start, 2)} second(s)')