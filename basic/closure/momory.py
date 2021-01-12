import time

# Memorization 는 동일한 입력이 다시 발생할 때 컴퓨터 프로그램의 속도를 높이기 위해 
# 값 비싼 함수 호출의 결과를 캐싱하는 기술입니다.  

# 일반적인 함수
def expensive_func(num):
    print("Computing {}...".format(num))
    time.sleep(1)
    return num*num

result = expensive_func(4)
print('Address of result is {}'.format(id(result)))
print(result)
result = expensive_func(10)
print('Address of result is {}'.format(id(result)))
print(result)
result = expensive_func(4)
print('Address of result is {}'.format(id(result)))
print(result)
result = expensive_func(10)
print('Address of result is {}'.format(id(result)))
print(result)


#     
print("ef_cache======================")
ef_cache = {}

def cache_func(num):
    if num in ef_cache:
        return ef_cache[num]

    print("Computing {}...".format(num))
    time.sleep(1)
    result = num*num
    ef_cache[num] = result
    return result

result = cache_func(4)
print('Address of result is {}'.format(id(result)))
print(result)
result = cache_func(10)
print('Address of result is {}'.format(id(result)))
print(result)
result = cache_func(4)
print('Address of result is {}'.format(id(result)))
print(result)
result = cache_func(10)
print('Address of result is {}'.format(id(result)))
print(result)

