import concurrent.futures
import time

def do_something(seconds=1):
  print(f'Sleeping {seconds} second(s)...')
  time.sleep(seconds)
  return 'Done Sleeping...'

if __name__ == '__main__':
  start = time.perf_counter()

  with concurrent.futures.ProcessPoolExecutor() as executor:
    p1 = executor.submit(do_something, 1)
    p2 = executor.submit(do_something, 1)
    
    print(p1.result())
    print(p2.result())

  # processes = []
  # for _ in range(10):
  #   p = multiprocessing.Process(target=do_something)
  #   p.start()
  #   processes.append(p)

  # for process in processes:
  #   process.join()

  finish = time.perf_counter()
  print(f'Finished in {round(finish-start, 2)} second(s)')

  