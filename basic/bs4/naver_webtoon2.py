# 네이버 웹툰 썸네일 가져오기
from bs4 import BeautifulSoup
from pprint import pprint
import requests, re, os
# 다운로드하기
from urllib.request import urlretrieve 

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
    #제목+썸내일 영역 추출
    li_list.extend(data1.findAll('li')) #해당 부분을 찾아 li_list와 병합

pprint(li_list)

# li 요소를 보면 img 태그에 썸네일 이미지의 주소와 웹툰 제목이 속성값으로 존재합니다. 
# img 태그를 추출한 뒤 속성값을 추출해봅시다. 추출한 데이터에 ['속성명']을 적으면 됩니다.

# 저장 폴더 생성
try:
    if not (os.path.isdir('images/naver_webtoon')):
        os.makedirs(os.path.join('images/naver_webtoon'))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("폴더 생성 실패!")
        exit()


# 각각 썸네일과 제목 추출하기
# 다운로드하기
for li in li_list:
    img = li.find('img')
    title = img['title']
    img_src = img['src']
    print(title,img_src)
    # urlretrieve( img_src , title+'.jpg') #주소, 파일경로+파일명+확장자
    title = re.sub('[^0-9a-zA-Zㄱ-힗]', '', title) # 해당 영역의 글자가 아니 것은 ''로 치환시킨다.
    urlretrieve( img_src , 'images/naver_webtoon/'+title+'.jpg') # 주소, 파일경로+파일명+확장자







