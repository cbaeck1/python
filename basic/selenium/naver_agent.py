# agent_crawling/agent_crawling.py
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

url = "https://land.naver.com/article/divisionInfo.nhn?rletTypeCd=A01&tradeTypeCd=&hscpTypeCd=A01%3AA03%3AA04&cortarNo=1171000000"

chrome_options = webdriver.ChromeOptions(); 
chrome_options.add_experimental_option("excludeSwitches", ['enable-logging']);
driver = webdriver.Chrome(options=chrome_options);  
# 암묵적으로 웹 자원 로드를 위해 3초까지 기다려 준다.
driver.implicitly_wait(3)

driver.get(url)

## 정지 버튼 누르기
driver.find_element_by_css_selector('a.btn_stop_on').click()

## 몇 명인지 확인
num = driver.find_element_by_css_selector('span.pagenum').text
num = int(num.split('/')[-1])
print("개수 :",num)

## 해당 영역 찾기
print("[정보 추출하기]")
results = []
for i in range(num):
   # " 문자열 ".strip() : 앞뒤 문자열 공백을 없애준다.
   data = {}
   profile = driver.find_element_by_css_selector('div.bx_com') # 프로필 영역

   ## 업체명과 대표명, 가능언어 추출
   data['company'] = profile.find_element_by_css_selector('h5.t_mem').text # 업체명
   data['owner'] = profile.find_element_by_css_selector('ul.lst_mem > li:nth-child(1)').text # 대표명 or 대표명|가능언어
   try: # "가능언어"가 있을 수도 있고 없을 수도 있으니
       lang_area = profile.find_element_by_css_selector('span.lang_area').text # 가능 언어
   except:
       lang_area = None

   ## 가능언어가 있을 때는 "대표명" = "대표명|가능언어" - "|가능언어"을 처리
   if lang_area : # lang_area 가 None 이면 False 처리된다.
       data['owner'] = data['owner'][:1+len(lang_area)] # "대표명" = "대표명|가능언어" - "|가능언어"
       data['langs'] = profile.find_elements_by_css_selector('span.lang_area > span.lang') # [영어 가능] or [영어, 중국어 가능] 의 요소
       ## 가능 글자 삭제하기
       index = 0
       while index < len(data['langs']):
           if ' 가능' in data['langs'][index].text:
               data['langs'][index] = data['langs'][index].text[:-3]
           else:
               data['langs'][index] = data['langs'][index].text
           index+=1

   ## 대표 글자 지우기
   if '대표 ' in data['owner'] : # 대표공백
       data['owner'] = data['owner'][len('대표 '):]

   ## 연락처 추출
   data['phones'] = profile.find_element_by_css_selector('ul.lst_mem > li:nth-child(2)').text # 전화 02-111-2222 / 010-2222-3333
   ## 전화 글자 삭제
   if '전화 ' in data['phones']: # 전화공백
       data['phones'] = data['phones'][len('전화 '):]
   data['phones'] = data['phones'].split(' / ') # 공백/공백
   print(data)

   # results에 추가
   results.append(data)

   # 더보기 클릭
   try:
       profile.find_element_by_css_selector('a.btn_next_on').click()
       time.sleep(.05) # 컴퓨터와 인터넷 상황이 좋다면 없애거나 숫자를 낮춰도 됩니다.
   except:
       break # 맨 마지막은 더보기 버튼이 클릭이 안된다.

## 엑셀 저장
print(["전체 댓글 엑셀로 저장"])
excel_name = "부동산 중개사"
import pandas as pd #pandas, openpyxl
data_frame = pd.DataFrame(results)
data_frame.to_excel('{}.xlsx'.format(excel_name),sheet_name='{}'.format(excel_name),startrow=0,header=True)