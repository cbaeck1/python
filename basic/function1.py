'''
12. 함수 알아보기
1. 함수
2. 파이썬 함수의 구조
def <함수명>(입력 인수 또는 인자):
    <수행할 문장1>
    <수행할 문장2>
    ...
3. 함수의 4가지 유형
3-1. 일반적인 함수(입력값과 출력값이 존재하는 함수)
3-2. 입력값이 없는 함수
3-3. 결과값이 없는 함수
3-4. 입력값과 결과값 모두 없는 함수
4. 입력값의 개수를 모를 때
def <함수형>(*입력 변수):
5. 함수의 입력값에 초기값 설정하기
6. 함수 안에서 선언한 변수의 효력 범위
7. 함수 안에서 함수 밖의 변수를 변경하는 방법


'''


# 4. 입력값의 개수를 모를 때
def add_many(*args):
    result = 0
    for i in args:
        result = result + i
    return result

result = add_many(1, 2, 3)
print(result)

# 키워드 파라미터 kwargs
# 키워드 파라미터를 사용할 때는 매개변수 앞에 별 두 개(**)를 붙인다.
# 딕셔너리로 만들어져서 출력
def print_kwargs(**kwargs):
    print(kwargs)

# {'a': 1}
print_kwargs(a=1)
# {'age': 3, 'name': 'foo'}
print_kwargs(name='foo', age=3)

def add_and_mul(a, b):
    return a+b, a*b

result1, result2 = add_and_mul(3, 4)

# 5. 함수의 입력값에 초기값 설정하기
def say_myself(name, old, man=True):
    print("나의 이름은 %s 입니다." % name)
    print("나이는 %d살입니다." % old)
    if man:
        print("남자입니다.")
    else:
        print("여자입니다.")

say_myself("박응용", 27)
say_myself("박응용", 27, True)

# 초깃값을 설정한 매개변수의 위치이다. 결론을 미리 말하면 이것은 함수를 실행할 때 오류가 발생한다.
# def say_myself(name, man=True, old):
# 매개변수로 (name, old, man=True)는 되지만 (name, man=True, old)는 안 된다는 것

# 6. 함수 안에서 선언한 변수의 효력 범위
a = 1

def vartest(a):
    a = a + 1  # 함수 안에서 새로 만든 매개변수는 함수 안에서만 사용하는 "함수만의 변수"이다

vartest(a)
print(a)  # 1

# 7. 함수 안에서 함수 밖의 변수를 변경하는 방법
# 1) return 사용하기
a = 1

def vartest1(a):
    a = a + 1
    return a

a = vartest1(a)
print(a)

# 2) global 명령어 사용하기
a = 1

def vartest2():
    global a
    a = a+1

vartest2()
print(a)

# lambda
# lambda 매개변수1, 매개변수2, ... : 매개변수를 이용한 표현식
def add(a, b): return a+b

result = add(3, 4)
print(result)
