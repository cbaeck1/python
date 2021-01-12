import time
import threading
import concurrent.futures

def do_something(seconds=1):
  print(f'Sleeping {seconds} second(s)...')
  time.sleep(seconds)
  return 'Done Sleeping...'

# 1. Using threading
start = time.perf_counter()
threads = []
for _ in range(10):
  t = threading.Thread(target=do_something, args=[1.5])
  t.start()
  threads.append(t)

for thread in threads:
  thread.join()

finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')

start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor() as executor:
  f1 = executor.submit(do_something, 1)
  f2 = executor.submit(do_something, 1)
  print(f1.result())
  print(f2.result())

finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')

# 2. Using concurrent & sumbit
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor() as executor:
  results = [executor.submit(do_something, 1) for _ in range(10)]
  for f in concurrent.futures.as_completed(results):  
    print(f.result())

finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')

start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor() as executor:
  secs = [5,4,3,2,1]
  results = [executor.submit(do_something, sec) for sec in secs]
  for f in concurrent.futures.as_completed(results):  
    print(f.result())

finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')

# 3. Using concurrent & map
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor() as executor:
  secs = [5,4,3,2,1]
  results = executor.map(do_something, secs)
  for result in results:
    print(result)

finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')



