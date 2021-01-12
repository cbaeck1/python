# 클로저(closure)는 내부함수와 밀접한 관계를 가지고 있는 주제다. 
# 내부함수는 외부함수의 지역변수에 접근 할 수 있는데 외부함수의 실행이 끝나서 외부함수가 소멸된 이후에도 
# 내부함수가 외부함수의 변수에 접근 할 수 있다. 이러한 메커니즘을 클로저라고 한다

# 스코프는 함수를 호출할 때가 아니라 함수를 어디에 선언하였는지에 따라 결정된다. 
# 이를 렉시컬 스코핑(Lexical scoping)라 한다
# 클로저는 함수와 그 함수가 선언됐을 때의 렉시컬 환경(Lexical environment)과의 조합이다.

# 클로저는 반환된 내부함수가 자신이 선언됐을 때의 환경(Lexical environment)인 스코프를 기억하여 
# 자신이 선언됐을 때의 환경(스코프) 밖에서 호출되어도 그 환경(스코프)에 접근할 수 있는 함수를 말한다. 
# 이를 조금 더 간단히 말하면 클로저는 자신이 생성될 때의 환경(Lexical environment)을 기억하는 함수다라고 말할 수 있겠다.

# 인자가 없는 함수, return 이 함수
def outer_function():
  message = 'Hi'

  def inner_function():
    print(message)
  
  return inner_function()

outer_function() # Hi

# return 이 함수이므로 함수를 변수로 선언하면 변수명만으로 함수가 실행됨
my_function = outer_function()
print("my_function")
my_function # 
# 변수를 함수호출하면 Exception has occurred: TypeError 'NoneType' object is not callable
# my_function()

# return 이 함수명
def outer_func():
  message = 'Hi'

  def inner_func():
    print(message)
  
  return inner_func

my_func = outer_func()
print("my_func.__name__", my_func.__name__)

# return 이 함수명이므로 함수를 변수로 선언하면 변수명만으로 살행되지 않는다
my_func
my_func()

# 인자가 있는 
def outer_f(msg):
  message = msg

  def inner_f():
    print(message)
  
  return inner_f

hi_func = outer_f('Hi!')
hello_func = outer_f('Hello!')

hi_func()
hello_func()

  