import time
import threading

def do_something():
  print('Sleeping 1 second...')
  time.sleep(1)
  print('Done Sleeping...')

start = time.perf_counter()
threads = []
for _ in range(10):
  t = threading.Thread(target=do_something)
  t.start()
  threads.append(t)

for thread in threads:
  thread.join()

finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')