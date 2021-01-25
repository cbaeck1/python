from selenium import webdriver
import time

browser = webdriver.Chrome()
url = "http://www.naver.com"

# 암묵적으로 웹 자원 로드를 위해 3초까지 기다려 준다.
browser.implicitly_wait(3)

# 1. 네이버 이동
browser.get(url)

# 스크린샷 찍기
browser.save_screenshot('data/naver1.png')
with open("data/navaer1.html", "w", encoding="utf8") as f:
    f.write(browser.page_source)

import requests
from bs4 import BeautifulSoup

soup = BeautifulSoup(browser.page_source, "lxml")

with open("data/navaer2.html", "w", encoding="utf8") as f:
    f.write(soup.prettify())

print(soup.a["href"])  # a element 의 href 속성 '값' 정보를 출력

mails = soup.find_all("i", attrs={"class":"ico_mail"})
print(mails)

browser.find_element_by_xpath('//*[@id="NM_FAVORITE"]/div[1]/ul[1]/li[1]/a/i').click()
time.sleep(1)
# 스크린샷 찍기
browser.save_screenshot('data/naver2.png')








