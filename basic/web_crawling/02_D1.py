'''
D. 색맹테스트 봇

1. 색맹테스트 직접해보기
    http://zzzscore.com/color/

2. 웹 페이지 코드 분석
게임에서 버튼이 4 ->9 -> 16.. 식으로 늘어나지만, 이미 div가 생성되있는 것을 개발자 도구를 통해서 알 수 있습니다.
각 버튼(div)의 xpath의 패턴을 분석해보면
(1,1)위치 div의 xpath : //*[@id="grid"]/div[1]
(1,2)위치 div의 xpath : //*[@id="grid"]/div[2]
(2,1)위치 div의 xpath : //*[@id="grid"]/div[3]
...
이런 식으로 되있습니다. 그렇다면 1 to 50 때 처럼 똑같이
//*[@id="grid"]/div[*] 로 xpath탐색하면 될 것 같은데, 그렇지가 않습니다.
* 은 찾고자 하는 것에 내부요소가 있는 경우만 탐색합니다.
그런데 이번에는 <div></div>로 태그 사이에 아무런 요소가 없습니다. 이 경우는 다음처럼 xpath를 탐색하겠습니다.
//*[@id="grid"]/div

3. 정답 탐색하기
1) 각 div의 rgba값 추출
2) 추출한 rgba중 값 하나 다른 곳이 정답

4. 디자인 정보(css) 추출
selenium 에서 추출한 요소는 value_of_css_property 로 해당 요소의 css속성을 확인할 수 있습니다. css란 html 요소의 디자인 속성 정보입니다.
모든 버튼에 대해 배경색상 정보를 추출하여 리스트에 저장합시다.

5. 정답 찾기
버튼들의 색상 정보를 가진 btns_rgba에서 값을 어떻게 찾아야 할까요? 여러 방법이 있겠지만 여기서는 collections모듈의 Counter함수를 이용해봅시다. 
Counter함수는 값(value)과 개수를 딕셔너리(dict)로 반환해줍니다.
반환된 딕셔너리에서 value가 1인 것이 곳 정답의 배경색 정보인 것이죠.

6. 정답 누르기
위에서 구한 배경색 정보를 토대로 정답 버튼을 클릭해봅시다.
버튼 요소를 저장한 btns에 배경색 정보를 순서대로 추출한 것이 btns_rgba 임으로 인덱스가 일치합니다. 따라서 다음 과정으로 정답을 클릭해봅시다.
1) answer에 저장된 색이 어느 위치(인덱스)에 있는지 btns_rgba에서 찾고
2) 그 인덱스로 btns에 저장된 요소를 조회하여 클릭을 한다.

7. 스톱워치

'''
from selenium import webdriver
from pprint import pprint
import time
from collections import Counter
from datetime import datetime, timedelta

driver = webdriver.Chrome()
driver.get('http://zzzscore.com/color/')
driver.implicitly_wait(3)

def clickBtn():
    btns = driver.find_elements_by_xpath('//*[@id="grid"]/div')
    # print(len(btns))

    # 3. 정답 탐색하기
    btns_rgba = [ btn.value_of_css_property('background-color') for btn in btns ]
    # pprint(btns_rgba)

    # 4. 정답 찾기
    result = Counter(btns_rgba)
    # pprint(result) #여기서 value가 1인게 정답

    # value가 1인 것 탐색
    answer = None
    for key, value in result.items():
        if value == 1:
            answer = key
            break
    else:
        answer = None
        print("정답을 찾을 수 없습니다.")

    # pprint(answer) 

    # 정답 클릭
    # 1. btns_rgba에서 인덱스 값을 구하고
    # 2. 그 인덱스 값으로 btns 인덱스에 접근. 클릭
    if answer :
        index = btns_rgba.index(answer)
        btns[index].click()
        return

# 7. 스톱워치 : 1분

period = timedelta(minutes=1)
next_time = datetime.now() + period
minutes = 0
while run == 'start':
    clickBtn()
    if next_time <= datetime.now():
        minutes += 1
        next_time += period


