
# 프로그래밍 언어가 함수를 First-class citizen으로 취급하는 것을 말한다.
# 함수를 First-class citizen으로 취급 가능하다고 하는 것은 다음을 뜻한다.
#   1. 함수를 변수나 자료구조에 저장할 수 있다.
#   2. 함수의 매개변수(인자)에 다른 함수명을 인수로 전달할 수 있다.
#   3. 함수의 반환값(return값)으로 함수명을 전달할 수 있다.
# 함수가 일급 함수인 대표적 언어 : Python, Go, Javascript, Kotlin
# 함수가 일급 함수가 아닌 대표적 언어 : C, Java

def square(x):
  return x * x

f = square(5)
print(square) # <function square at 0x00000193DD595820>
print(f)      # 25

f = square
print(square) # <function square at 0x00000193DD595820>
print(f)      # <function square at 0x00000193DD595820>
print(f(5))   # 25

def cube(x):
  return x * x * x

def my_map(func, arg_list):
  result = []
  for i in arg_list:
    result.append(func(i))
  return result

squares = my_map(square, [1,2,3,4,5])  
print(squares)

cubes = my_map(cube, [1,2,3,4,5])  
print(cubes)


def logger(msg):
  def log_message():
    print('Log:', msg)
  return log_message

log_hi = logger('Hi!')  
log_hi()

def logger2(msg):
  def log_message():
    print('Log:', msg)
  return log_message()

logger2('Hello!')

def html_tag(tag):
  def wrap_text(msg):
    print('<{0}>{1}</{0}'.format(tag, msg))
  return wrap_text

print_h1 = html_tag('h1')  
print_h1('Test Headline!')
print_h1('Another Headline!')

print_p = html_tag('p')
print_p('Test Paragraph!')



