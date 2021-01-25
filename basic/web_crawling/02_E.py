'''
E. 트위치 클립 다운로드

1. 클립 영상 소스링크
트위치 클립은 <video>태그에 src속성을 확인하면됩니다. 그동안은 xpath로만 탐색을 했는데, 
이번에는 추출할 요소의 태그가 명확하니 find_element_by_tag_name 를 사용하겠습니다.
selenium에서 추출한 요소의 속성값을 확인하려면 get_attribute 를 사용할 수 있습니다.
2. 영상 제목과 날짜
영상 링크는 구했으니, 제목과 날짜를 추출해봅시다. 나중에 이 2가지를 이용하여 파일명으로 만듭시다.
3. 영상 다운로드
막상 다운로드하려니 걸리는 것이 있습니다. 바로 특수문자입니다. 
01.D. 네이버 웹툰 썸네일 가져오기 때처럼 특수문자삭제 처리를 해주고, 빈칸도 _(언더바)로 대신 바꿔보겠습니다.
다운로드는 urlretrieve 을 사용


'''

from selenium import webdriver
import time

driver = webdriver.Chrome()
# 1. 특정 클립 링크
url = 'https://www.twitch.tv/hanryang1125'
driver.get(url) 
time.sleep(3)

# video 태그 확인
url_element = driver.find_element_by_tag_name('video')
vid_url = url_element.get_attribute('src')
print(vid_url)

# 2. 영상 제목과 날짜
title_element1 = driver.find_element_by_class_name('tw-flex')
title_element2 = title_element1.find_elements_by_tag_name('span')
vid_title,vid_date = None, None
for span in title_element2:
    try:
        d_type =span.get_attribute('data-test-selector')
        if d_type == "title":
            vid_title = span.text
        elif d_type == 'date':
            vid_date = span.text
    except:
        pass

print(vid_title,'\t',vid_date)

# 3. 특수문자 없애고 빈칸도 없에기
import re
vid_title = re.sub('[^0-9a-zA-Zㄱ-힗]', '', vid_title)
vid_date = re.sub('[^0-9a-zA-Zㄱ-힗]', '', vid_date)
print(vid_title,'\t',vid_date)

from urllib.request import urlretrieve
urlretrieve(vid_url, 'images/'+vid_title+'_'+vid_date+'.mp4')

driver.close()