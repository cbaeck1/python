'''
파이썬 클린코드

#1_ Docstring

0. 문서화
Docstring에 대해서 알아보기전에 문서화에 대해서 한번 짚어보자.
프로젝트를 진행해보거나, 다른 개발자와 함께 협업을 진행해본 개발자라면 '문서화'가 왜 필요한지 어느정도 느꼈을 수 있다.
여러분들은 지금 개발하고 있는 프로젝트, 지금 작성한 코드를 한달, 일년뒤에 보았을 때 코드를 작성할 때와 같이 
부드럽게 코드리딩이 가능한가? 
사실 한달, 일년도 아니다. 복잡한 로직과 다양한 처리를 진행하는 개발을 진행할 때면 
당장 내일에 그 코드를 정확히 기억하기 힘든 경우도 많을 것이다.
물론 그렇게 리딩이 어렵다는 것은 로직 자체가 깔끔하지 못하다는 문제이지 않을까란 생각을 할 수도 있을 수 있다. 
당연히 그 또한 중요하겠지만, 만약 내가 작성한 코드를 다른 동료 개발자가 본다면 어떠할까? 
로직이 깔끔하지만 수많은 함수와 클래스들의 코드를 직접 리딩해가며 모든 로직을 '코드'만으로 이해하고 받아들인다는건 
상상하는 것보다 매우 힘든 일이 될 수 있다.
하지만 만약 각각의 함수와 클래스, 기타 등등이 어떤 기능을 하는 것인지 알아볼 수 있도록 '문서화'를 해두었다면 어떨까? 
내가 작성한 코드를 매우 오랜만에 보더라도 해당 함수의 매개변수는 어떤 타입인지, 해당 함수가 어떤 기능을 하는 것인지, 
그리고 반환 값은 무엇인지 한번에 알아 볼 수 있어 해당 함수를 이용하거나 받아들이는데 매우 편리할 것이다. 
즉, 문서화를 통해서 구현된 코드에 대해 보다 명확하고 편리하게 설명할 수 있으며 특정 함수나 클래스 등을 
다른 곳에서 사용하고자 할 때 
그것들에 대해 손쉽고 명확하게 이해하고 사용할 수 있어 추가적인 개발에 있어 사전에 버그를 방지할 수 있을 것이다.
물론 위에서 이야기한 내용보다 '문서화'가 필요하고 중요한 이유는 더 다양하고 더 중요한 이유들이 있을 수 있지만 
이정도라면 개발자 누구나 '문서화'가 왜 필요한지 어느정도 스스로 생각해 볼 수 있는 계기가 될 것이라 생각한다.

1. Docstring 이란?
그럼 왜 필자는 Docstring에 대한 이야기에 앞서 문서화를 이야기했을까?
Docstring은 쉽게 생각했을 때, 코드에 포함된 문서(Document)이다. 즉, 코드에 대한 문서화를 코드 밖에, 
워드나 한글 파일, 엑셀을 이용해 따로 하는 것이 아니라 코드 내부에 문서화를 실시한다는 것이다.
특히나 파이썬과 같은 동적 타입의 언어이기 때문에 코드내부에 문서를 포함시키는 docstring 이 매우 좋다. 
파이썬에서는 파라미터의 타입을 체크하거나 강제하지 않는다. 헌데 특정 함수나 클래스를 이용하거나 수정하고자 할때 
그에 대한 설명이 명확하게 나와있다면 그러한 이용이나 수정이 매우 간편하게 진행될 것이다.
파이썬에서 docstring은 함수나 클래스 모듈등에 정의할 수 있다. 
그리고 작성한 내용을 확인하기 위해서는 해당 객체의 __doc__ 라는 속성을 이용하면 된다. (더블언더바)
docstring에 대한 사용법을 알아보기 이전에 실제로 docstring이 정의된 예시를 확인해보자.
print(len.__doc__)
print(dict.__doc__)
위와 같이 dictionary 개체의 docstring을 보니, 어떻게 사용하는지도 설명을 첨부하였다.
만약 우리가 dictionary가 무엇인지 궁금했다면 이와 같이 docstring을 통해서 간략하게나마 확인할 수 있을 것이다.

2. Docstring 사용하기
그럼 직접 우리가 클래스나 함수에 대해서 Docstring을 만들어서 사용해보도록 하자.

class DocstringClassExample():
    """
    DocstringClassExample() 예제 클래스
    class에 대한 설명을 함께 적어준다.
    """
 
    def docstring_func_example():
        """
        Return 0 always
        """
        print("ocstring_func_example 함수를 실행하였습니다.")
        return 0
 
def main():
    print("Docstring 만들어보기")
    new_doc = DocstringClassExample()
    print("Class docstring start")
    print(new_doc.__doc__)
    print("Class docstring end")
    print()
    print("Function docstring start")
    print(new_doc.docstring_func_example.__doc__)
    print("Function docstring end")
 
 
if __name__ == '__main__':
    main()

위와 같이 간단한 Class 하나와 함수하나를 만들었다.
docstring은 위와 같이 클래스나 함수 선언 바로 하단부에 따옴표 세개를 이용하면 된다.(쌍따옴표, 홀따옴표 무관하다.)
이렇게 만든 코드를 실행시켜보면 다음과 같이 결과가 출력된다.
Docstring 만들어보기
Class docstring start

        DocstringClassExample() 예제 클래스
        class에 대한 설명을 함께 적어준다.

Class docstring end

Function docstring start

                Return 0 always

Function docstring end

docstring 자체를 만들어내는 것, 이를 사용하는 것은 함께 알아본 것 처럼 전혀 어렵지 않고 너무 간단하다.
하지만 보다 중요한 것은 이러한 기능을 놓치지 않고 이용하면서 보다 유지보수 좋은 코드를 지속적으로 만들어 나가는 것, 
그리고 기존에 있던 docstring을 최신으로 유지하는 노력일 것이다.


#2_ Annotation

1. Annotation 이란?
우선 Annotation의 사전적 정의는 "주석"이다. 즉, 쉽게 말해서 코드에 대한 추가적인 설명을 이야기하는 무언가를 의미한다.
대표적으로 Java언어에서 함수나 클래스 상단에 @를 통해 annotation을 표시한다.
위는 java의 spring 프레임워크를 사용한 코드 중 일부인데, 10번, 13번, 16번 라인을 보면 
@를 통해 annotation을 사용하고 있음을 볼 수 있다. 물론 자바에서의 annotation과 
파이썬에서의 annotation은 차이가 있을 수 있다. 
하지만 기본적으로 로직이 되는 코드를 "도와주는" 의미에서는 크게 다르지 않다고 볼 수 있다.
(사실 java의 annotation과 같은 것을 파이썬에서는 decorator로 나타내기 때문에 엄밀히 말했을 때, 
자바의 annotation과 파이썬의 annotation은 다르다고 볼 수 있다.)
보다 자세히, 파이썬에서의 annotation에 대해서 알아보자.
사실 파이썬에서는 annotation에 대한 강제성이 전혀 없다. 우리가 파이썬에서 #을 이용하거나, 
지난 포스팅에서 알아본 docstring과 같이 안써도 되지만, 보다 좋은 코드가 될 수 있도록 
추가적으로 관리해주는 것 중 하나일 뿐이다.
파이썬에서 사용하는 annotation의 기본 아이디어는, 코드를 사용하는 이에게 함수나 클래스에 대해 
그 인자값의 형태 또는 반환값을 알려주자는 것이다.
함수에 대해서는 함수의 인자에 대한 타입에 대한 힌트를 적거나,
함수의 return값에 대한 타입을 알려줄 수 있다. 또한 파이썬 3.6이후로는 변수에 대해서도 직접 annotation을 달 수 있다. 
즉, 클래스에서 사용되는 변수값에 대해 그 타입을 적어둘 수 있다는 것이다.

2. Annotation 사용하기
#-*- coding:utf-8 -*-
class AnnotationClassExample:
    """
    Annotation에 대한 예시를 확인하기 위한 class입니다.
    __annotation__ 속성을 통해
    class할당되는 first_param과 second_param에 대한 타입을 확인할 수 있습니다.
    """
    first_param: str
    second_param: int
 
    def set_first_param(self, value: str) -> None:
        """
        AnnotationClassExample 클래스의
        first_param 값을 바인딩합니다.
        함수의 반환은 없습니다.
        """
        self.first_param = value
 
    def set_second_param(self, value: int) -> bool:
        """
        AnnotationClassExample 클래스의
        second_param 값을 바인딩합니다.
        함수의 반환은 True or False 입니다.
        """
        if type(value) == int:
            self.second_param = value
            return True
        else:
            self.second_param = 0
            return False
 
def main():
    print("Annotation 만들어보기")
    new_class = AnnotationClassExample()
    print("\n* AnnotationClassExample 클래스의 annotations")
    print(new_class.__annotations__)
    print("\n* set_first_param 함수의 annotations")
    print(new_class.set_first_param.__annotations__)
    print("\n* set_second_param 함수의 annotations")
    print(new_class.set_second_param.__annotations__)
 
if __name__ == '__main__':
    main()

위의 코드에서는 AnnotationClassExample 클래스와 그 내부에 2개의 변수를 가지고 있으며, 
클래스 내부에 2가지 함수를 추가로 구현해두었다. 
우선 함수에 대한 annotation을 살펴보자.
11번, 19번 라인의 함수선언부를 살펴보면 함수의 인자에 대한 annotation과 함수의 return에 대한 annotation이 적용되었다.
이를 통해 함수를 사용하고자 하는 이는 함수의 인자가 어떤 타입을 가져야하는지, 
그리고 함수를 통해 얻게되는 값의 타입은 무엇인지 보다 쉽게 알 수 있다.
또한 파이썬 3.6부터 변수에 대한 annotation이 가능하다고 했는데, 
이는 8번, 9번 라인과 같이 클래스 내부의 변수에 대한 annotation으로 사용할 수 있다. 
이렇게 annotation을 적용하면, 그 개체에 대해 __annotations__ 이라는 속성이 생긴다. 
그리고 해당 속성을 통해 우리가 적용해둔 annotation 값을 볼 수 있는 것이다.
실제로 위의 코드를 실행시켜 보면 main함수내에서 각 클래스와 함수에 대해 __annotations__ 속성을 호출하고, 
그 결과는 다음과 같다.

Annotation 만들어보기

* AnnotationClassExample 클래스의 annotations
{'first_param': <class 'str'>, 'second_param': <class 'int'>}

* set_first_param 함수의 annotations
{'value': <class 'str'>, 'return': None}

* set_second_param 함수의 annotations
{'value': <class 'int'>, 'return': <class 'bool'>}

우리가 코드에서 적용시켜준 annotation들이 출력되는 것을 확인할 수 있다.
위와 같이 annotation을 통해 함수나 변수 등에 미리 타입에 대한 힌트를 적어둘 수 있다.
물론 이 또한 파이썬에서 강제성이 있거나, 지켜야 한다는 것은 아니다. annotation은 말 그대로 '힌트'를 주는 것에 불과하다.


#3_ 개발 지침 약어

1. DRY / OAOO
DRY(Do not Repeat Yourself)와 OAOO(Once And Only Once)는 강조하고자 하는 의미가 비슷하므로 함께 다루어보자. 
두개의 약어는, '중복을 피하라'라는 의미를 가지고 있다.
즉, 특정 기능과 역할을 하는 것은 코드에 단 한 곳에 정의되어 있어야 하고 중복되지 않아야 한다. 
그리고 이를 통해 코드를 변경하고자 할 때 수정이 필요한 곳은 단 한 군데만 존재해야 한다.
코드의 중복이 발생한다는 건 유지보수를 하는데에 있어서 직접적인 영향을 미칠 수 있다는 것이다. 
다양한 문제가 있을 수 있지만 축약해보면 다음과 같은 3가지 문제가 대표적이다.

- 오류가 발생하기 쉽다.
특정 계산 로직이 코드 전체 여러곳에 동일하게 분포되어 있을 때, 계산 로직에 대한 변경사항이 발생하면 코드의 모든 곳을 찾아 
변경해주어야 하는데 이때 하나라도 빠뜨리면 오류가 발생하기 쉬워진다.
- 비용이 발생한다.
동일한 기능에 대한 반복 수정이 이루어져야 하기 때문에, 당연히 1회의 수정보다 다수의 수정에 있어서 비용적으로 손해가 발생한다.
- 신뢰성이 떨어진다.
동일한 기능이 코드 여러 곳에 분포되어 있을 때, 모든 곳을 찾아서 수정해야 한다. 물론 언어적 기능과 도구의 도움을 받을 수도 있지만, 
모든 곳을 정확히 기억하지 못할 수 있다는 점 때문에 시스템 전체의 신뢰성이 보다 떨어질 수 있다.
간단하게 나마 코드의 중복이 발생할 수 있는 예시와 적절히 조치 된 예시를 살펴보자.

#-*- coding:utf-8 -*-
# DRY / OAOO
 
user_math_score_dic = {
    'A': 90,
    'B': 93,
    'C': 30,
    'D': 100,
    'E': 31,
    'F': 82,
    'G': 79,
}
 
user_eng_score_dic = {
    'A': 30,
    'B': 63,
    'C': 39,
    'D': 94,
    'E': 10,
    'F': 49,
    'G': 68,
}
 
# Danger code
def get_user_score_list(user_math_score_dic, user_eng_score_dic):
    """
    input: 유저의 이름을 key로, 점수를 value로 가지는 dict형 자료형 2개
    output: 종합 점수 계산에 따라 내림차순으로 정렬한 유저의 이름 list
    """
    
    user_sum_score_dic = {}
    # 종합 점수 계산 (math*2 + eng)
    for k, math_score in user_math_score_dic.items():
        sum_score = math_score*2
        sum_score += user_eng_score_dic[k]
        user_sum_score_dic[k] = sum_score
 
    # 종합 점수에 따라 내림차순 정렬
    sorted_user = sorted(user_sum_score_dic.keys(), key=lambda x: user_sum_score_dic[x])
    return sorted_user
 
print("# Danger code")
print(get_user_score_list(user_math_score_dic, user_eng_score_dic))
 
 
# Good code
def calc_user_sum_score(user_math_score_dic, user_eng_score_dic):
    """
    input: 유저의 이름을 key로, 점수를 value로 가지는 dict형 자료형 2개
    output: 종합 점수 계산이 된 dict 자료형
    """
    user_sum_score_dic = {}
    # 종합 점수 계산 (math*2 + eng)
    for k, math_score in user_math_score_dic.items():
        sum_score = math_score*2
        sum_score += user_eng_score_dic[k]
        user_sum_score_dic[k] = sum_score
    return user_sum_score_dic
 
def get_user_score_list2(user_math_score_dic, user_eng_score_dic):
    """
    input: 유저의 이름을 key로, 점수를 value로 가지는 dict형 자료형 2개
    output: 종합 점수 계산에 따라 내림차순으로 정렬한 유저의 이름 list
    """
    user_sum_score_dic = calc_user_sum_score(user_math_score_dic, user_eng_score_dic)
 
    # 종합 점수에 따라 내림차순 정렬
    sorted_user = sorted(user_sum_score_dic.keys(), key=lambda x: user_sum_score_dic[x])
    return sorted_user
 
print("# Good code")
print(get_user_score_list(user_math_score_dic, user_eng_score_dic))
 

위의 코드를 보면 기존에 정의된 get_user_score_list 함수에서는 내부적으로 종합 점수에 대한 계산이 진행되고 있다.
만약 그러한 계산 로직이 다른 곳에서도 필요하면 어떻게 될까? 따로 함수화가 되어 있지 않기 때문에 동일 로직을 중복시켜야 한다. 
하지만 아래와 같이 calc_user_sum_score 라는 함수를 분리해두면, 추후 동일 로직이 필요할 때 해당 함수를 이용할 수 있을 것이다.

2. YAGNI / KIS
YAGNI(You Aren't Gonna Need It)와 KIS(Keep It Simple) 또한 의미하는 바가 비슷하므로 함께 다루도록 하자. 
두 약어가 의미하는 것은 '현재 주어진 문제에 적합한, 간단한 코드를 작성하라'이다.
YAGNI에서 보다 강조하는 것은, 과잉된 프로그래밍을 하지 말라는 것이다. 
우리는 결론적으로 시스템에 대한 확장성과 유지보수 등을 위해 보다 좋은 코드를 작성하려고 한다. 
하지만 그 목표가 코드를 작성하는 시점에 특정 미래적 상황을 예측해야 한다는 것은 아니다. 
필자가 참고하고 있는 서적에서는 이렇게 이야기 한다.
유지보수가 가능한 소프트웨어를 만드는 것은 미래의 요구 사항을 예측하는 것이 아니다.
- 파이썬 클린 코드

위에서의 말대로, 우리가 확장성과 유지보수 등을 위한 소프트웨어를 만들어야 한다는 것은, 다가오지 않은 미래에 대해 
특정 상황을 예측해야 한다는 것은 아니다. 오히려 그랬다면 코드적인 학습보다, 미래학자와 같은 학습을 해야하지 않을까 싶다. 
따라서, 프로그래밍 시점에서는 현재의 요구사항을 잘 해결하기 위한 소프트웨어를 작성하되, 이때에 보다 수정가능하고, 
높은 응집도와 낮은 결합력을 가지는 프로그래밍을 해야한다. 미래에 ~이 필요할거야, 나중에 ~가 고려되지 않을까, 라는 생각에 
현재의 요구사항을 넘어서는, 과잉 프로그래밍을 하면 안된다.
KIS에서 조금 더 강조하는 점은 현재에 선택한 솔루션이 최선의, 최소한의 솔루션이어야 한다는 것이다. 
문제를 해결하는데에 있어서 화려하고 어려운 기술은 필수요소가 아니다. 항상 보다 간결하고 최소한의 솔루션으로 문제를 해결해야 한다.
단순하게 해결될 수 있는 문제를 보다 복잡하게 해결하게 되면 추후 해당하는 함수, 클래스, 데이터에 대한 수정에 있어 
더 큰 어려움이 내포될 수 있다.
더군다나, 파이썬의 철학에서는 '단순한 것이 복잡한 것보다 낫다.' 라고 이야기 하고 있다.


3. EAFP / LBYL
EAFP(Easier to Ask Forgiveness than Permission)와 LBYL(Look Before You Leap)는 상대적인 의미를 지니고 있는 약어이다.
우선 EAFP는 허락보다 용서를 구하는 것이 쉽다는 말인데 이 의미는 일단 코드가 실행되도록 두고 동작하지 않을 경우를 
대응한다는 의미이다. 일반적으로는 코드가 실행되도로 하고 발생할 수 있는 에러에 대해서 catch, except문을 이용해 
조치하는 코드를 의미한다.
이에 반해 LBYL은 도약하기 전에 살피라는 말이며, 의미적으로는 코드가 실행되기 이전에 확인/검토를 하라는 의미이다. 
간단하게는 if문 등을 이용한 체크 정도로 생각하면 된다.
아래 코드는 파일을 사용하기 이전에 LBYL에 따른 코드와, EAFP에 따른 코드를 나타내고 있다.

# LBYL
if os.path.exists(filename):
    with open(filename) as f:
        ...

# EAFP
try:
    with open(filename) as f:
        ...
        
except FileNotFoundError as e:
    logger.error(e)
    ...
 







'''