'''
14. 클래스(Class)란
1. 클래스(class)는 틀, 공장과 같은 역할을 해서 자신의 형태와 같은 결과물(인스턴스)를 만들어 냅니다.
2. 클래스(Class)에서 self 와 __init__
__init__ : 인스턴스가 생성될 때 항상 실행되는 것 : 생성자 (Constructor)
self : 생성되는 객체 자신
  객체 자신을 통해서 해당 함수를 호출하면 self를 생략할 수 있습니다.
  Adder(adder_2, 3) == adder_2(3)
3. 클래스의 상속 : 어떤 클래스를 만들 때 다른 클래스의 기능을 물려받을 수 있게 만드는 것
class 클래스 이름(상속할 클래스 이름)
기존 클래스를 변경하지 않고 기능을 추가하거나 기존 기능을 변경하려고 할 때 사용
4. 메서드 오버라이딩 : 부모 클래스(상속한 클래스)에 있는 메서드를 동일한 이름으로 다시 만드는 것
5. 클래스 변수
Family 클래스로 만든 객체를 통해서도 클래스 변수를 사용할 수 있다.
클래스 변수는 클래스로 만든 모든 객체에 공유된다

'''
result = 0

def adder(num):
    global result
    result += num
    return result

print(adder(3))
print(adder(6))
print(adder(12))

# 특정 상황에 의해, adder라는 계산기가 두개 필요
# 예를 들어, 3+4+5 라는 연산과 9+10 이라는 연산을 한번에 진행
result_1 = 0

def adder_1(num):
    global result_1
    result_1 += num
    return result_1

result_2 = 0

def adder_2(num):
    global result_2
    result_2 += num
    return result_2

print(adder_1(3))
print(adder_1(6))
print(adder_1(12))

print(adder_2(9))
print(adder_2(10))


class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result

# 별개의 계산기 adder_1, adder_2, adder_3 (파이썬에서는 이것을 객체라고 부른다)
# 계산기(cal1, cal2)의 결괏값 역시 다른 계산기의 결괏값과 상관없이 독립적인 값을 유지
# 클래스를 사용하면 계산기 대수가 늘어나더라도 객체를 생성만 하면 되기 때문에 함수를 사용하는 경우와 달리 매우 간단해진다.
# 만약 빼기 기능을 더하려면 Calculator 클래스에 다음과 같은 빼기 기능 함수를 추가해 주면 된다
adder_1 = Calculator()
adder_2 = Calculator()
adder_3 = Calculator()

print(adder_1, adder_2, adder_3)
adder_1.add(3)

print(adder_1.add(3))
print(adder_1.add(6))
print(adder_1.add(12))
print(adder_2.add(9))
print(adder_3.add(10))

# 사칙연산 클래스 만들기
# 객체에 숫자 지정할 수 있게 만들기 :  setdata
#   setdata 메서드의 매개변수
#   setdata 메서드의 수행문
# 기능 만들기 : 더하기, 곱하기, 빼기, 나누기
# 생성자 (Constructor)
class FourCal:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def setdata(self, first, second):
        self.first = first
        self.second = second

    def add(self):
        result = self.first + self.second
        return result

    def mul(self):
        result = self.first * self.second
        return result

    def sub(self):
        result = self.first - self.second
        return result

    def div(self):
        result = self.first / self.second
        return result

a = FourCal(4, 2)
print(a)
print(a.add())


# 상속 개념을 사용하여 우리가 만든 FourCal 클래스에 ab (a의 b제곱)을 구할 수 있는 기능을 추가
class MoreFourCal(FourCal):
    def pow(self):
        result = self.first ** self.second
        return result

# div 메서드를 호출하면 4를 0으로 나누려고 하기 때문에 위와 같은 ZeroDivisionError 오류가 발생
class SafeFourCal(FourCal):
    def div(self):
        if self.second == 0:  # 나누는 값이 0인 경우 0을 리턴하도록 수정
            return 0
        else:
            return self.first / self.second

# 클래스 변수
class Family:
    lastname = "김"

print(Family.lastname)

# Family 클래스로 만든 객체를 통해서도 클래스 변수를 사용할 수 있다.
# 클래스 변수는 클래스로 만든 모든 객체에 공유된다
a = Family()
b = Family()

print(a.lastname)
print(b.lastname)
