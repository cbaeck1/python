# 데코레이터
# 자신의 방을 예쁜 벽지나 커튼으로 장식을 하듯이, 기존의 코드에 여러가지 기능을 추가하는 방식


def outer_function():
  message = 'Hi'

  def inner_function():
    print(message)
  
  return inner_function()

outer_function()

def outer_function(msg):
  message = msg

  def inner_function():
    print(message)
  
  return inner_function

hi_func = outer_function('Hi')
bye_func = outer_function('Bye')

hi_func()
bye_func()



def decorator_function(original_function):  
    def wrapper_function():  
        return original_function()  
    return wrapper_function  

def display():  
    print('display 함수가 실행됐습니다.') 

decorated_display = decorator_function(display)  
decorated_display()  

