# -*- coding: utf-8 -*-
from __future__ import division
import os
import psutil
import random
import time

names = ['최용호', '지길정', '진영욱', '김세훈', '오세훈', '김민우']
majors = ['컴퓨터 공학', '국문학', '영문학', '수학', '정치']

def people_list(num_people):
    result = []
    # ython 3 에서는 range() 와 xrange() 가 통합되어 range() 만 제공되며, 그 특성은 xrange() 와 동일하다.
    # for i in xrange(num_people):
    for i in range(num_people):
        person = {
            'id': i,
            'name': random.choice(names),
            'major': random.choice(majors)
        }
        result.append(person)
    return result

def people_generator(num_people):
    for i in range(num_people):
        person = {
            'id': i,
            'name': random.choice(names),
            'major': random.choice(majors)
        }
        yield person

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024

# AttributeError: module 'time' has no attribute 'clock'
# t1 = time.clock()
t11 = time.perf_counter() # time.process_time()
people = people_list(10000000)  #1 people_list 를 호출
t12 = time.perf_counter()
mem_after = process.memory_info().rss / 1024 / 1024
total_time = t12 - t11

print('list ----------------------')
print('시작 전 메모리 사용량: {} MB'.format(mem_before))
print('종료 후 메모리 사용량: {} MB'.format(mem_after))
print('총 소요된 시간: {:.6f} 초'.format(total_time))
time.sleep(5)


t21 = time.perf_counter() # time.process_time()
people = people_generator(10000000)  # people_generator 를 호출
t22 = time.perf_counter()
mem_after = process.memory_info().rss / 1024 / 1024
total_time = t22 - t21

print('yield ----------------------')
print('시작 전 메모리 사용량: {} MB'.format(mem_before))
print('종료 후 메모리 사용량: {} MB'.format(mem_after))
print('총 소요된 시간: {:.6f} 초'.format(total_time))



