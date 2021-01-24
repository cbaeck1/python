# 월요 웹툰 제목 추출
# 실습 주소는 여기입니다. https://comic.naver.com/webtoon/weekday.nhn 먼저 월요일 웹툰의 제목만 추출을 해봅시다.
from bs4 import BeautifulSoup
from pprint import pprint
import requests

#웹 페이지를 열고 소스코드를 읽어오는 작업
html = requests.get("http://comic.naver.com/webtoon/weekday.nhn")
soup = BeautifulSoup(html.text, 'html.parser')
html.close()

# 월요웹툰영역 추출하기
# 월요 웹툰 영역을 소개하는 곳은 div 태그에 class "col_inner" 속성과 값을 가지고 있습니다.
# data1=soup.find('div',{'class':'col_inner'})
# pprint(data1)

# 제목 영역 코드 추출하기
# <a class="title" href="/webtoon/list.nhn?titleId=716857&amp;weekday=mon" onclick="nclk_v2(event,'thm*m.tit','','23')" title="웹툰제목">
# 웹툰제목
# </a>
# data2=data1.findAll('a',{'class':'title'})
# pprint(data2)

# 텍스트만 추출 1
# title_list = []
# for t in data2:
#     title_list.append(t.text)

# 텍스트만 추출 2
# title_list = [t.text for t in data2]
# pprint(title_list)

# 모든 요일 웹툰 제목 가져오기
# [월요웹툰영역find] - [해당영역 제목 findAll] - [for로 text 추출]
# 이제 [월요웹툰영역find] 부분을 [요일별 웹툰영역 findAll] 로 변경시킬 겁니다. 일주일이니 7개가 되겠죠.
# 즉 뒤에 따라오는 [해당영역 제목 findAll] - [for로 text 추출] 부분도 각 7번씩 수행되어야 합니다.
# 요일별 웹툰영역 추출하기
data1_list=soup.findAll('div',{'class':'col_inner'})
# pprint(data1_list)

# 전체 웹툰 리스트
week_title_list = []
for data1 in data1_list:
    #제목 포함영역 추출하기
    data2=data1.findAll('a',{'class':'title'})
    # pprint(data2)

    #텍스트만 추출 2
    title_list = [t.text for t in data2]
    pprint(title_list)

    week_title_list.extend(title_list) #단순하게 값을 추가해 1차원으로 만들려면 extend
    # week_title_list.append(title_list) #요일별로 나눠 2차원 리스트를 만들려면 append













