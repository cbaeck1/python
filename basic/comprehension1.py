'''
23_ 컴프리헨션(Comprehension) 이란

1. 컴프리헨션(Comprehension)이란?
사전적으로는 이해, 이해력, 포용, 포용력, 포함, 압축 등의 뜻을 가지고 있습니다.
앞으로 알아보는 Comprehension을 보다 제대로 이해하기 위해서는 기본적으로 파이썬의 조건문, 반복문 등의 개념을 알고 있으셔야 하며 
해당 개념은 List, Set, Dict 자료형에 대해 사용될 수 있기 때문에, 
해당 자료형에 대해서도 알아야 제대로 이해할 수 있습니다.

2. List, Set, Dict Comprehension
먼저 알아볼 내용은 List, Set, Dict 자료형으로 사용되는 Comprehension입니다.
사용되는 방법이나, 문법등은 동일하나 어떤 자료형에 적용시키는지에 따라 다름
List Comprehension 은 반복되거나 특정 조건을 만족하는 리스트를 보다 쉽게 만들어 내기 위한 방법입니다.
2-1. 반복문을 사용한 Comprehension
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print([i for i in range(10)])
# [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
print([i * 3 for i in range(10)])
# [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print([i + 2 for i in range(10)])

리스트를 생성하기 위해서 직접 값을 넣어주거나, 기타 다른 함수들을 이용했지만 Comprehension이라는 개념을 사용하면 
보다 쉽게 리스트를 생성할 수 있습니다.

2-2. 조건문을 사용한 Comprehension
반복문과 조건문을 이용하여 리스트를 생성
먼저 반복문을 적어주고, 반복문의 인자가 뒤에 적어준 조건문에 해당하는지를 확인하여 그 인자를 리스트의 요소로 가지게 되는 것입니다.
또한, 위에서 사용한 반복문이나 조건문을 여러개 이용하는 것도 가능합니다.
이때, 여러번 사용되는 반복문은 각각이 개별적으로 돌아가는 것이 아니고 조건문은 서로 and연산으로 묶인다고 생각하시면 됩니다.
아래 예시들을 통해 확인해보도록 하겠습니다.





2-3. 두 개의 반복문 사용하기



먼저, 두개의 반복문이 사용된 예제입니다.







먼저 반복문이 어떻게 돌아가는지 보다 잘 확인하기 위해 a,b,c,d,e 를 가진 리스트 a와 1,2,3,4,5를 가진 리스트 b를 만들어 주었습니다.

그리고 [i+j for i in a for j in b] 라는 Comprehension 구문으로 새로운 리스트를 생성해보았습니다.

해당 결과를 보면 먼저 앞의 for문에서 하나의 요소에 대해 뒤의 for문을 적용하는 방식임을 확인할 수 있습니다.

보다 쉽게 보자면, 아래와 같은 것임을 알 수 있습니다.







위의 코드를 보시면, 우리가 전에 알던 for문의 문법을 이용하여 a, b 리스트에 있는 요소를 하나씩 꺼내어 새로운 리스트에 삽입하는 방식으로 결과를 확인해보았습니다.





2-4. 두 개의 조건문 사용하기



이번에는 두개의 조건문이 사용된 예제를 확인해보도록 하겠습니다.







위의 결과를 보시면 0~49의 범위에서 2로 나눴을 때 0이며, 3으로 나눴을 때 0인 요소로 새로운 리스트가 생성된 것을 확인할 수 있습니다.

즉, 우리가 적어준 두개의 조건문이 서로 and로 묶인 것을 알 수 있습니다.





2-5. 조건문에서 else 사용하기



물론 우리가 사용하는 조건문에 대해서 if 이외에 else도 함께 사용할 수 있습니다. 하지만 else if (elif)는 사용할 수 없습니다.







위의 코드의 첫번째 줄에서 볼 수 있듯이 if와 else를 함께 사용할 수 있습니다.

0~9의 숫자에 따라서 2로 나누어 떨어지면 'even'을, 그게 아니면 'odd'를 출력하게 한 구문 입니다.

하지만 그 아래줄에서 0인 숫자에 대해서는 zero를 출력하도록, elif를 사용해보았지만 오류가 발생합니다.



그렇지만, else를 사용할 수 있기때문에 elif를 직접적으로 사용할 수는 없지만, 개념적으로 elif와 동일한 구문을 사용할 수는 있습니다.





2-6. 조건문에서 elif 사용하기







위와 같이 else 뒤에서 if를 한번 더 사용함으로써, elif와 같은 기능을 갖는 구문을 만들 수 있습니다.

첫 번째 줄에서는 0이 2로 나누어떨어진다고 보기 때문에 맨 앞에 있는 if 문에 걸려 even을 출력하게 되었습니다. 이를 수정하여 두번째 코드와 같이 작성하면 우리가 기대한 값이 출력됨을 볼 수 있습니다.

물론 이러한 경우에도 else를 여러번 중복하여 사용할 수 있습니다.







위의 코드에서는 0~9의 숫자에 대해서, 1일때는 'one', 2일때는 'two', 3일때는 'three', 4일때는 'four'를 출력하도록 여러개의 else와 if를 사용하였습니다. 그리고 그외에는 'hum'을 출력하도록 설정하였습니다.





3. Generator Expression



이번에는 Generator Expression에 대해서 알아봅니다.

지금까지 Comprehension에 대해서 알아보다가 갑자기 다른 주제라서 당황하셨나요?

하지만 Generator Expression도 Comprehension를 사용한 기능 중에 하나입니다.



처음에, Comprehension도 리스트말고도 {}를 사용하는 집합 자료형과 키와 값을 이용한 딕셔너리 자료형에서도 이용할 수 있다고 말씀드렸습니다.

그리고 이와 비슷하게, ()를 사용하면 Generator Expression이 되는 것 입니다.

표현하는 방법 자체는 위에서 알아본 Comprehension과 동일하니 어떻게 사용되는 것인지만 예제를 통해서 알아보도록 하겠습니다.







먼저 a와 b에 우리가 위에서 알아본 Comprehension 의 구문과 소괄호, ()를 이용하여 정의하였습니다. 그리고 각각을 print해보니 generator 객체가 나왔습니다.

이처럼 Comprehension 구문을 사용하여 소괄호로 묶어주면 자동적으로 파이썬에서 generator expression으로 인식하여 generator 객체를 생성하게 됩니다.

이를 사용하기 위해서는 해당 generator객체를 next로 감싸서 출력해주면 됩니다.



우리가 Comprehension를 통해 기대하는 값들이 순서대로 하나씩 출력되는 것을 볼 수 있습니다. 그리고, 모든 값에 대해 한바퀴를 돌게 되면 아래와 같이 Stop Iteration 이라는 오류를 출력하게 됩니다.









4. Comprehension 정리
우리는 위에서 Comprehension에 대해서 알아보았습니다.
그 구문과 사용법을 익히면, list, set, dict과 같은 자료형에서 다양하게 사용될 수 있으며, 소괄호를 이용하여 generator 객체를 만들어 이용할 수도 있습니다.
이러한 구문이 나왔을 때 당황하지 않고 리딩할 수 있어야 하며, 필요할때는 적극적으로 이용할 수 있어야 하기 때문에 꼭 한번씩 인터프리터등을 이용해 연습해보시기를 추천드립니다.
마지막으로 Comprehension 구문에 대해서 총정리를 해보면, 아래와 같습니다.




'''

# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print([i for i in range(10)])
# [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
print([i * 3 for i in range(10)])
# [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print([i + 2 for i in range(10)])

# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
print([i for i in range(30) if i%2 == 0])
# [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
print([i for i in range(30) if i%3 == 0])

a = ['a','b','c','d','e']
b = ['1','2','3','4','5']
# ['a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4', 'b5', 'c1', 'c2', 'c3', 'c4', 'c5', 'd1', 'd2', 'd3', 'd4', 'd5', 'e1', 'e2', 'e3', 'e4', 'e5']
print([i+j for i in a for j in b])

new_list = []
for i in a:
    for j in b:
        new_list.append(i+j)
print(new_list)

# [0, 6, 12, 18, 24, 30, 36, 42, 48]
print([i for i in range(50) if i%2 == 0 if i%3 == 0])

# ['even', 'odd', 'even', 'odd', 'even', 'odd', 'even', 'odd', 'even', 'odd']
print(['even' if i%2 == 0 else 'odd' for i in range(10)])

# ['zero', 'odd', 'even', 'odd', 'even', 'odd', 'even', 'odd', 'even', 'odd']
print(['zero' if i == 0 else 'even' if i%2 == 0 else 'odd' for i in range(10)])

# ['hum', 'one', 'two', 'three', 'hum', 'hum', 'hum', 'hum', 'hum', 'hum']
print(['one' if i == 1 else 'two' if i == 2 else 'three' if i == 3 else 'hum' for i in range(10)])


a = (i+1000 for i in range(10))
b = (i+9999 for i in range(10))

print('next a is ', next(a), '|| next b is ', next(b))
print('next a is ', next(a), '|| next b is ', next(b))
print('next a is ', next(a), '|| next b is ', next(b))
print('next a is ', next(a), '|| next b is ', next(b))
print('next a is ', next(a), '|| next b is ', next(b))
print('next a is ', next(a), '|| next b is ', next(b))
print('next a is ', next(a), '|| next b is ', next(b))
print('next a is ', next(a), '|| next b is ', next(b))
print('next a is ', next(a), '|| next b is ', next(b))
print('next a is ', next(a), '|| next b is ', next(b))
# Exception has occurred: StopIteration
# print('next a is ', next(a), '|| next b is ', next(b))

