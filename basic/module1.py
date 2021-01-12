'''
15. 모듈
모듈이란 함수나 변수 또는 클래스를 모아 놓은 파일

1. 모듈 만들기
2. 모듈 불러오기 :  import는 현재 디렉터리에 있는 파일이나 파이썬 라이브러리가 저장된 디렉터리에 있는 모듈만 불러올 수 있다.
import 모듈이름
from 모듈이름 import 모듈함수
3. if __name__ == "__main__": 의 의미
3.1 import mod1을 수행하는 순간 mod1.py가 실행이 되어 결괏값을 출력
3.2 __name__ 변수란? 파이썬의 __name__ 변수는 파이썬이 내부적으로 사용하는 특별한 변수 이름이다. 
직접 mod1.py 파일을 실행할 경우 mod1.py의 __name__ 변수에는 __main__ 값이 저장된다. 
파이썬 셸이나 다른 파이썬 모듈에서 mod1을 import 할 경우에는 mod1.py의 __name__ 변수에는 모듈 이름 값 mod1이 저장된다.

4. 클래스나 변수 등을 포함한 모듈
5. 다른 파일에서 모듈 불러오기






'''
# 파이썬은 main문이 없는 대신에, 들여쓰기가 되지 않은 Level0의 코드를 가장 먼저 실행
# Run import
import module_wrong
import mod1_wname
import mod1_woname
import module_main
print("Second Module's name: {}".format(__name__))

# print(add(1, 4)), print(sub(4, 2))
#
print("Second Module's name: {}".format(__name__))


# First Module's name: first_module_wrong
print("Second Module's name: {}".format(__name__))
