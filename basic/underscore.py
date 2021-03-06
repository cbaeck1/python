# 파이썬 언더스코어(_)에 대하여
# 1. 인터프리터에서 사용되는 경우
# 2. 값을 무시하고 싶은 경우
# 3. 특별한 의미의 네이밍을 하는 경우
# 4. 국제화(i18n)/지역화(l10n) 함수로 사용되는 경우
# 5. 숫자 리터럴값의 자릿수 구분을 위한 구분자로써 사용할 때

# 1. 인터프리터에서 사용되는 경우
# 파이썬 인터프리터에선 마지막으로 실행된 결과값이 _라는 변수에 저장
a=10
_=10
print(a, _, _*3, _*20)

# 2. 값을 무시하고 싶은 경우
# 언패킹시 특정값을 무시
x, _, y = (1, 2, 3) # x = 1, y = 3

# 여러개의 값 무시
x, *_, y = (1, 2, 3, 4, 5) # x = 1, y = 5

def do_something():
    pass

# 인덱스 무시
for _ in range(10):
    do_something()


list_of_tuple = [(1,2)]
print(type(list_of_tuple), type((1,2)))
# 특정 위치의 값 무시
for _, val in list_of_tuple:
    do_something()


# 3. 특별한 의미의 네이밍을 하는 경우


# 4. 국제화(i18n)/지역화(l10n) 함수로 사용되는 경우


# 5. 숫자 리터럴값의 자릿수 구분을 위한 구분자로써 사용할 때