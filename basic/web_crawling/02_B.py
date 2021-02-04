'''
B. 유튜브 키워드 검색

1. 검색 키워드 자동입력
유튜브의 키워드 검색창의 경로를 알아봅시다. selenium에서는 대체로 xpath로 경로를 계산하여 요소에 탐색합니다. 
먼저 크롬 개발자 도구로 검색창의 xpath를 알아내봅시다.
//*[@id="search"]

2. Enter 전송
엔터나 방향키같이 특수한 키는 다음과 같이 입력할 수 있습니다.
1) from selenium.webdriver.common.keys import Keys 을 참조 
2) search.send_keys(Keys.원하는키) 를 실행
예를 들어 엔터(enter)를 전송하고 싶으면 Keys.Enter 입니다. 가능한 키는
Keys.ARROW_DOWN , Keys.ARROW_LEFT , Keys.ARROW_RIGHT
Keys.ARROW_UP , Keys.BACK_SPACE , Keys.CONTROL
Keys.ALT , Keys.DELETE , Keys.ENTER , Keys.SHIFT
Keys.SPACE , Keys.TAB , Keys.EQUALS , Keys.ESCAPE
Keys.HOME , Keys.INSERT , PgUp Key  Keys.PAGE_UP
Keys.PAGE_DOWN , Keys.F1 , Keys.F2 , Keys.F3 , Keys.F4
Keys.F5 , Keys.F6 , Keys.F7 , Keys.F8 , Keys.F9 , Keys.F10
Keys.F11 , Keys.F12

'''

from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys

# driver = webdriver.Chrome('chromedriver')
driver = webdriver.Chrome()
driver.get("https://www.youtube.com/")
time.sleep(3)

# 1. 검색 키워드 자동입력
# 검색어 창을 찾아 search 변수에 저장
search = driver.find_element_by_xpath('//*[@id="search"]')

# search 변수에 저장된 곳에 값을 전송
search.send_keys('반원 코딩')
time.sleep(1)

# 2. Enter 전송
search.send_keys(Keys.ENTER)


