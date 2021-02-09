'''
13. 사용자 입출력, 파일IO
1. 사용자 입력
2. 출력
3. 파일 IO(Input / Output)
파일 객체 = open(파일 경로 및 이름, 파일 열기 모드)
r : 읽기모드, 파일을 읽기만 할 때 사용
w : 쓰기모드, 파일에 내용을 쓸 때 사용
a : 추가모드, 파일의 마지막에 새로운 내용을 추가할 때 사용


1. 파일 생성하기
2. 파일을 쓰기 모드로 열어 출력값 적기
3. 프로그램의 외부에 저장된 파일을 읽는 여러 가지 방법
3.1 readline() 함수 이용하기
3.2 readlines 함수 사용하기
3.3 read 함수 사용하기
4. 파일에 새로운 내용 추가하기
5. with문과 함께 사용하기


'''

# 1. 파일 생성하기
import sys
f = open("새파일.txt", 'w')
for i in range(1, 11):
    data = "%d번째 줄입니다.\n" % i
    f.write(data)
f.close()


# 3. 프로그램의 외부에 저장된 파일을 읽는 여러 가지 방법
f = open("새파일.txt", 'r')
line = f.readline()
print(line)
f.close()


f = open("새파일.txt", 'r')
while True:
    line = f.readline()
    if not line:
        break
    print(line)
f.close()


f = open("새파일.txt", 'r')
lines = f.readlines()
for line in lines:
    print(line)
f.close()


f = open("새파일.txt", 'r')
data = f.read()
print(data)
f.close()


# 4. 파일에 새로운 내용 추가하기
f = open("C:/doit/새파일.txt", 'a')
for i in range(11, 20):
    data = "%d번째 줄입니다.\n" % i
    f.write(data)
f.close()


# 5. with문과 함께 사용하기
# with문을 사용하면 with 블록을 벗어나는 순간 열린 파일 객체 f가 자동으로 close
with open("foo.txt", "w") as f:
    f.write("Life is too short, you need python")

with open("foo.txt", "r") as f:
    print(f.readline(), end="")

# [sys 모듈로 매개변수 주기]
# 명령 프롬프트 명령어 [인수1 인수2 ...]
# 파이썬에서는 sys 모듈을 사용하여 매개변수를 직접 줄 수있다.
# #sys 모듈을 사용하려면 아래 예의 import sys처럼 import 명령어를 사용해야 한다.


args = sys.argv[1:]
for i in args:
    print(i)
