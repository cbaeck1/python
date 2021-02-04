'''
D. 네이버 웹툰 썸네일 가져오기

1. 제목과 썸네일이 같이 존재하는 영역
요일 웹툰영역의 li 태그 안에는 제목과 썸네일이 포함되어있습니다.
2. 제목과 이미지 링크 추출
li 요소를 보면 img 태그에 썸네일 이미지의 주소와 웹툰 제목이 속성값으로 존재합니다. 
img 태그를 추출한 뒤 속성값을 추출해봅시다. 추출한 데이터에 ['속성명']을 적으면 됩니다.
3. 다운로드하기
이미지 또는 동영상 링크가 있다면 다운로드하는 방법은 쉽습니다.
from urllib.request import urlretrieve 를 추가한 뒤, urlretrieve 호출 시에 링크와 저장할 파열명을 넣으면 됩니다. 
4. 특수문자 처리
도중에 에러가 난 부분을 보면 파일명에 특수문자가 있는 경우입니다.
따라서 추출한 제목에서 특수문자는 다른 문자로 변경해주거나 삭제를 해주도록 하겠습니다. 
변경은 replace를 하면 되는데, 여기서는 정규식 표현을 이용한 re모듈을 사용하여 삭제해주도록 하겠습니다. 
따라서 re모듈을 import 해주세요. 
5. 저장 폴더 생성
현재 파이썬 파일과 동일 위치에 저장되니 관리가 힘듭니다. 자동으로 image폴더를 생성하고 그곳에 저장시키도록 합시다.
여기서는 os모듈을 참조하고 아래 함수들을 사용하겠습니다. 
os.path.isdir : 이미 디렉토리가 있는지 검사 
os.path.join : 현재 경로를 계산하여 입력으로 들어온 텍스트를 합하여 새로운 경로를 만듬 
os.makedirs : 입력으로 들어온 경로로 폴더를 생성
모듈 참조와 아래 urlretrieve 부분도 변경을 해주세요.

'''

from bs4 import BeautifulSoup
from pprint import pprint
import requests, re, os
from urllib.request import urlretrieve

# 5. #저장 폴더를 생성
try:
    if not (os.path.isdir('images/naver_webtoon')):
        os.makedirs(os.path.join('images/naver_webtoon'))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("폴더 생성 실패!")
        exit()


# 웹 페이지를 열고 소스코드를 읽어오는 작업
html = requests.get("http://comic.naver.com/webtoon/weekday.nhn")
soup = BeautifulSoup(html.text, 'html.parser')
html.close()

# 요일별 웹툰영역 추출하기
data1_list=soup.findAll('div',{'class':'col_inner'})
# pprint(data1_list)

# 전체 웹툰 리스트
li_list = []
for data1 in data1_list:
    # 제목+썸내일 영역 추출
    li_list.extend(data1.findAll('li')) #해당 부분을 찾아 li_list와 병합
# pprint(li_list)

# 2. 제목과 이미지 링크 추출
for li in li_list:
    img = li.find('img')
    title = img['title']
    img_src = img['src']
    # print(title, img_src)
    # 4. 특수문자 처리 : 해당 영역의 글자가 아니 것은 ''로 치환시킨다
    title = re.sub('[^0-9a-zA-Zㄱ-힗]', '', title)  
    # 3. 다운로드하기 : 주소, 파일경로+파일명+확장자
    urlretrieve(img_src , 'images/naver_webtoon/'+title+'.jpg')  

    