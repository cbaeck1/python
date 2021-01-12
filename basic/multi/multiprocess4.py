import concurrent.futures
import multiprocessing
import time

def do_something(seconds=1):
  print(f'Sleeping {seconds} second(s)...')
  time.sleep(seconds)
  return f'Done Sleeping...{seconds}'

def main():
  try:
    # multiprocessing
    start = time.perf_counter()
    processes = []
    for _ in range(10):
      p = multiprocessing.Process(target=do_something)
      p.start()
      processes.append(p)

    for process in processes:
      process.join()
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    # concurrent submit
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
      p1 = executor.submit(do_something, 1.5)
      p2 = executor.submit(do_something, 2)
      
      print(p1.result())
      print(p2.result())

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    # concurrent
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
      results = [executor.submit(do_something, 1.5) for _ in range(10)]    
      for p in concurrent.futures.as_completed(results):  
        print(p.result())

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    # concurrent
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
      secs = [5,4,3,2,1]
      results = [executor.submit(do_something, sec) for sec in secs]    
      for p in concurrent.futures.as_completed(results):  
        print(p.result())

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    # concurrent map
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
      secs = [5,4,3,2,1]
      results = executor.map(do_something, secs)
      for result in results:
        print(result)
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

  except KeyboardInterrupt:
    print( "User aborted.")
    sys.exit()


# Main Entry
if __name__=="__main__":
  main()