'''
B. 네이버 날씨 미세먼지 가져오기

1. 웹 페이지 가져오기
네이버에 날씨 페이지를 이용하면, 요청장소의 위치를 알아서 계산해 주기때문에 GPS 정보를 따로 계산할 필요가 없습니다.
2. 파싱
웹 페이지는 HTML이라는 언어로 쓰여져있습니다. 이를 파이썬에서 쉽게 분석할 수 있도록 파싱작업을 거쳐 각 요소에 접근이 쉽게 만들겠습니다.
3. 크롬 개발자 도구
개발자 도구를 이용하면 현재 웹 페이지 요소를 쉽게 분석할 수 있습니다.
4. 요소 1개 찾기(find)
이제 각 요소에 쉽게 접근할 수 있습니다. 미세먼지 정보가 있는 div 요소만 추출해봅시다.
여기서 중요한 부분은 추출된 웹페이지 소스코드입니다.
HTML 요소는 <태그 속성 = 속성값> 텍스트 가 기본 구조입니다.(예외도 있습니다.)
즉 위 그림에서 알아낸 요소는
    태그 : div
    속성 : class
    속성값 : detail_box 입니다.
5.요소 모두 찾기(findAll)
find와 사용방법이 똑같으나 find 는 처음 매칭된 1개만, findAll 은 매칭된 모든 것을 반환하며 리스트로 반환합니다.
리스트 길이 3개가 확인되며, 0~2 인덱스에 해당되는 값은 각각 미세먼지, 초미세먼지, 오존지수 줄에 해당합니다.
6. 내부 텍스트 추출
가운데에 있는 실질적인 데이터(숫자와 단위) 부분만 추출해봅시다.
그런데 이렇게 하면 코드가 나와버립니다. 여기서는 내부 텍스트만 골라내도록 .text를 이용해봅시다. 
위에서 언급한 HTML의 기본 구성을 확인하세요. <태그>텍스트에서 텍스트만 골라내는 방법입니다.
7. 초미세먼지 추출
여태까지 다룬 것을 응용하여 초미세먼지도 출력해봅시다. data2 변수에서 미세먼지는 0번 인덱스, 초미세먼지는 1번 인덱스였다는 것을 떠올리세요.

'''

from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests

# 1. 웹 페이지 가져오기
html = requests.get('https://search.naver.com/search.naver?query=날씨')
#pprint(html.text)

# 2. 파싱
soup = bs(html.text,'html.parser')

# 4. 요소 1개 찾기(find)
data1 = soup.find('div',{'class':'detail_box'})
pprint(data1)

# 5.요소 모두 찾기(findAll)
data2 = data1.findAll('dd')
pprint(data2)

# 6. 내부 텍스트 추출
fine_dust = data2[0].find('span',{'class':'num'})
print(fine_dust)

# 7. 초미세먼지 추출
ultra_fine_dust = data2[1].find('span',{'class':'num'}).text
print(ultra_fine_dust)

# 8. 오존지수
o3 = data2[2].find('span',{'class':'num'}).text
print(o3)

