'''
C. 네이버 웹툰 제목 가져오기

1. 월요 웹툰 제목 추출
   https://comic.naver.com/webtoon/weekday.nhn 
   
2. 월요 웹툰 영역
월요 웹툰 영역을 소개하는 곳은 div 태그에 class "col_inner" 속성과 값을 가지고 있습니다.
2.1 제목 영역 코드 추출하기
<a class="title" href="/webtoon/list.nhn?titleId=716857&amp;weekday=mon" onclick="nclk_v2(event,'thm*m.tit','','23')" title="웹툰제목">
웹툰제목
</a>
여기서 findAll을 이용하여 a태그의 class="title"를 검사하면 제목 텍스트를 추출하는 준비가 됩니다.
2.2 for를 이용한 텍스트 추출 또는 리스트에서의 for문 사용법

3. 모든 요일 웹툰 제목 가져오기
[월요웹툰영역find] - [해당영역 제목 findAll] - [for로 text 추출]
이제 [월요웹툰영역find] 부분을 [요일별 웹툰영역 findAll] 로 변경시킬 겁니다. 일주일이니 7개가 되겠죠.
즉 뒤에 따라오는 [해당영역 제목 findAll] - [for로 text 추출] 부분도 각 7번씩 수행되어야 합니다.

3.1 요일별 영역 가져오기
위에서 다룬 코드와 비교하시면서 따라오시길 바랍니다. 기존 find부분을 findAll로 변경합니다.
그리고 변수명은 data1_list로 변경하겠습니다.(아래서 이렇게 한 이유에 대해 나옵니다.)

3.2 하나의 리스트로 묶기
현재 title_list는 1회성으로 사용되고 있습니다. 요일별 title_list가 생성될 때마다 특정 리스트에 저장하도록 작성


4. 단순하게 웹툰 제목을 가져오는 코드
'''


from bs4 import BeautifulSoup
from pprint import pprint
import requests

# 1. 웹 페이지를 열고 소스코드를 읽어오는 작업
html = requests.get("http://comic.naver.com/webtoon/weekday.nhn")
soup = BeautifulSoup(html.text, 'html.parser')
html.close()

# 2. 월요 웹툰 영역
data1=soup.find('div',{'class':'col_inner'})
pprint(data1)

# 2.1 제목 영역 코드 추출하기
data2=data1.findAll('a',{'class':'title'})
pprint(data2)

# 2.2 for를 이용한 텍스트 추출
title_list = []
for t in data2:
    title_list.append(t.text)
pprint(title_list)

# 2.3 리스트에서의 for문 사용법
title_list = [t.text for t in data2]
pprint(title_list)


# 3. 요일별 웹툰영역 추출하기
data1_list=soup.findAll('div',{'class':'col_inner'})
pprint(data1_list)

week_title_list = []
for data1 in data1_list:
    #제목 포함영역 추출하기
    data2=data1.findAll('a',{'class':'title'})

    #텍스트만 추출 2
    title_list = [t.text for t in data2]
    pprint(title_list)

    week_title_list.extend(title_list) # 단순하게 값을 추가해 1차원으로 만들려면 extend
    # week_title_list.append(title_list) # 요일별로 나눠 2차원 리스트를 만들려면 append

pprint(week_title_list)


# 4. 단순하게 웹툰 제목을 가져오는 코드
data1=soup.findAll('a',{'class':'title'})
week_title_list = [ t.text for t in data1]
pprint(week_title_list)

