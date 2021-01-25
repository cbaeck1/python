from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests

html = requests.get('https://search.naver.com/search.naver?query=날씨')
#pprint(html.text)

with open("data/mynaverweather.html", "w", encoding="utf8") as f:
    f.write(html.text)

# 웹 페이지 가져오기
soup = bs(html.text,'html.parser')

# 요소 1개 찾기(find)
# 여기서 중요한 부분은 추출된 웹페이지 소스코드입니다.
# HTML 요소는 <태그 속성 = 속성값> 텍스트 가 기본 구조입니다.(예외도 있습니다.)
# 즉 위 그림에서 알아낸 요소는
# 태그 : div
# 속성 : class
# 속성값 : detail_box 입니다.
# 이제 변환한 데이터에 find( 태그, { 속성 : 속성값} ) 를 사용하여 해당 부분만 추려봅시다.

data1 = soup.find('div',{'class':'detail_box'})
pprint("data1: "+ str(data1))

# 요소 모두 찾기(findAll)
# find와 사용방법이 똑같으나 find 는 처음 매칭된 1개만, findAll 은 매칭된 모든 것을 반환하며 리스트로 반환합니다.
# 개발자 도구를 이용하면 각 줄이 <dd>에 해당하는 걸 알 수 있습니다. 마침 미세먼지 줄이 1번째라 find를 써도 되지만, 
# 나중에 초미세먼지, 오존지수도 접근할 수 있도록 코드를 작성하겠습니다.
# find와 findAll은 사용방법이 같습니다. 즉 매개변수 부분도 같습니다.
# 그리고 find와 findAll 둘 다 태그만 가지고도 사용할 수 있습니다.(즉, 속성과 속성값 부분이 생략 가능합니다.)
# div.detail_box 까지 추출하여 data1 에 저장했으니, 여기에 findAll 을 사용해봅시다.

data2 = data1.findAll('dd')
pprint("data2: "+ str(data2))

# 리스트 길이 3개가 확인되며, 0~2 인덱스에 해당되는 값은 각각 미세먼지, 초미세먼지, 오존지수 줄에 해당합니다.

# 내부 텍스트 추출
# 가운데에 있는 실질적인 데이터(숫자와 단위) 부분만 추출해봅시다.
# span 태그에 속성과 속성값은 class = "num" 입니다.  

fine_dust = data2[0].find('span',{'class':'num'})
print("fine_dust: "+ str(fine_dust))

# 그런데 이렇게 하면 코드가 나와버립니다. 여기서는 내부 텍스트만 골라내도록 .text를 이용해봅시다. 
# 위에서 언급한 HTML의 기본 구성을 확인하세요. <태그>텍스트에서 텍스트만 골라내는 방법입니다.

# 초미세먼지 추출
# 여태까지 다룬 것을 응용하여 초미세먼지도 출력해봅시다. data2 변수에서 미세먼지는 0번 인덱스, 초미세먼지는 1번 인덱스였다는 것을 떠올리세요.

ultra_fine_dust = data2[1].find('span',{'class':'num'}).text
print("ultra_fine_dust: "+ str(ultra_fine_dust))
