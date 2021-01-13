'''
' _ ' 와 같은 표시를 언더스코어라고 하는데 이것이 파이썬에서는 보다 특별한 의미를 가지고 사용

1. 값을 무시하고 싶은 경우
제일 먼저 알아볼 경우는 값을 무시하고 싶은 경우입니다. 물론 쓰레기 변수를 만들어서 사용해도 되겠지만 언더스코어를 
사용하는 경우 더 깔끔하고, 코드를 확인할 때 사용하지 않는 값임을 나타내기 때문에 보다 직관적일 수 있다고 생각합니다.
사실상, 값을 무시하는 경우가 언제 필요할까? 라는 생각이 드시는 분도 있겠지만 간단하게는 아래와 같은 코드에서 값을 무시할 경우가 생깁니다.

for _ in range(5):
    print("인덱스는 필요없어")

def return_two_var(a):
    return a, a**2

_, double = return_two_var(5)
print(double)

우리가 for 문을 사용할때 지정되는 인덱스가 굳이 필요 없을때가 있습니다. 
이럴때 언더스코어를 사용하며 또는, 우리가 함수 등에서 한번에 두개의 값을 반환받을 때 둘 중 하나의 값만 이용하고 싶다면 사용할 수 있습니다.
즉, 우리가 원래 변수로 사용하던 곳에 대신하여 언더스코어( _ )를 사용하면 그 값은 무시하게 되는 것 입니다.

2. 숫자 자리수 구분을 하는 경우
숫자의 자리수 구분으로 사용하는 경우는 간단합니다. 
우리는 보통 1000 대신 1,000와 같이 반점을 이용해 숫자를 세자리씩 구분하는데 이러한 역할이라고 생각하면 됩니다.

print(1000)
print(1_000)
print(1000000)
print(1_000_000)

3. private 으로 선언하고 싶은 경우
이번에는 private 으로 선언하고 싶은 경우입니다. 물론 변수/메소드/클래스 등 모든 것을 포함하는데, 
사실상 이는 private 이라고 보기에는 약간 애매할 수 있습니다.
예들 들어, 아래와 같이 함수와 변수를 선언했다고 생각해보겠습니다.

def _hello():
    print('hello, python')

_private_var = "I'm private"

물론 이후에, _hello() 함수를 실행시키거나 _private_var 를 출력해도 결과가 올바르게 나옵니다. 
그럼 무엇때문에 private 선언이라고 하느냐?
예를 들어, 위의 코드를 hello.py 라는 이름으로 저장하고, 다른 파일에서 이를 불러온다고 생각해봅시다.
그럼 우리는 from hello import * 와 같은 식으로 모듈을 불러오게 됩니다.
하지만 이때 언더스코어( _ )로 시작된 함수나 클래스, 변수는 사용할 수 없게됩니다. 
사용할 수 없다기보다 불러와지지 않는다는 말이 맞을듯 합니다.
즉, 외부에서 모듈을 불러올때 언더스코어로 시작된 변수나 함수, 클래스는 사용하지 못하는 것입니다.
하지만 이 또한 사용할 수 있는 방법이 있기는 있습니다.
import 를 할때 * 를 사용해서 하는 것이 아니고, 직접 함수나 변수명을 언급하여 import를 하면 사용할 수 있게됩니다.
완벽하게 private 이라고 볼 수는 없겠지만, 숨기고 싶은 함수나 변수, 클래스를 만들때는 직관적으로라도 이해하기 위해 
언더스코어를 사용하여 네이밍하는 것이 좋을 것 같습니다.

4. 중복된 이름을 피하고 싶은경우
이는 맹글링을 사용한다고 표현하기도 합니다.
맹글링이란 프로그래밍 언어 자체적으로 가지고 있는 규칙에 의해서 함수나 변수의 이름을 변경하는 것을 말합니다. 
즉, 파이썬에서 가지고 있는 규칙에 의해서 우리가 네이밍한 변수나 함수의 이름을 변경하는 것 입니다.

class A:
    def function1(self):
        print('function1 of class A')
    def function2(self):
        print('function2 of class A')
class B:
    def function1(self):
        print('function1 of class B')
    def function2(self):
        print('function2 of class B')

a = A()
b = B()
a.function1()
b.function2()

A 클래스와 B 클래스의 두개의 함수명이 서로 같습니다. 물론 잘 구분해서 사용한다면 문제가 없겠지만, 
코드가 복잡해지고 변수이름이 다양해졌을때에는 충분히 헷갈릴 수 있게됩니다.
이럴때 각 클래스에 대해서 중복된 함수명을 우리가 만들더래도, 맹글링을 통해 알아서 구분되는 함수명으로 변경시키기 위해 
더블언더스코어( __ )를 사용합니다.

class A:
    def __function1(self):
        print('function1 of class A')
    def __function2(self):
        print('function2 of class A')
class B:
    def __function1(self):
        print('function1 of class B')
    def __function2(self):
        print('function2 of class B')

a = A()
b = B()
a._A__function1()
b._B__function2()      


함수를 네이밍할때 그 앞에 더블언더스코어( __ )를 사용하였습니다.
그리고 각 함수를 호출하기 위해서는 하단에서 보이는 것처럼 _<클래스명><함수명>과 같이 호출해야 합니다.
즉, 같은 함수명을 가졌더라도 함수를 호출할 때 직접적으로 클래스명을 언급해주어야 하므로 
코드를 이해할때 보다 직관적으로 이해할 수 있습니다.

+ 인터프리터에서 사용할 때
마지막으로 알아볼 것은 언더스코어를 인터프리터에서 사용할 때 입니다.
인터프리터에서는 마지막 변수의 값을 일시적으로 가지고 있는 용도로 사용됩니다.

>>> a = 10
>>> b = 20
>>> a + b
30
>>> _
30

'''


print(1000)
print(1_000)
print(1000000)
print(1_000_000)

def _hello():
    print('hello, python')

_private_var = "I'm private"


class A:
    def function1(self):
        print('function1 of class A')
    def function2(self):
        print('function2 of class A')
class B:
    def function1(self):
        print('function1 of class B')
    def function2(self):
        print('function2 of class B')

a = A()
b = B()
a.function1()
b.function2()

class A1:
    def __function1(self):
        print('function1 of class A1')
    def __function2(self):
        print('function2 of class A1')
class B1:
    def __function1(self):
        print('function1 of class B1')
    def __function2(self):
        print('function2 of class B1')

a = A1()
b = B1()
a._A1__function1()
b._B1__function2()      
