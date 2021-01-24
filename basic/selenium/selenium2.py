from selenium import webdriver
import time

options = webdriver.ChromeOptions() 
options.add_experimental_option("excludeSwitches", ["enable-logging"])

options = webdriver.ChromeOptions()
options.headless = False
options.add_argument("window-size=1920x1080")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36"
)

#browser = webdriver.Chrome(options=options, executable_path=r'chromedriver.exe')
browser = webdriver.Chrome(options=options)
# 암묵적으로 웹 자원 로드를 위해 3초까지 기다려 준다.
browser.implicitly_wait(3)

# 네이버 이동
browser.get('http://www.naver.com')

print(browser.page_source)
# 스크린샷 찍기
browser.save_screenshot('data/naver1.png')
with open("data/navaer1.html", "w", encoding="utf8") as f:
    f.write(browser.page_source)

import requests
from bs4 import BeautifulSoup

soup = BeautifulSoup(browser.page_source, "lxml")

with open("data/navaer2.html", "w", encoding="utf8") as f:
    f.write(soup.prettify())
    
time.sleep(1)
print(soup.a["href"])  # a element 의 href 속성 '값' 정보를 출력

mails = soup.find_all("i", attrs={"class":"ico_mail"})
print(mails)

browser.find_element_by_xpath('//*[@id="NM_FAVORITE"]/div[1]/ul[1]/li[1]/a/i').click()
time.sleep(1)
# 스크린샷 찍기
browser.save_screenshot('data/naver2.png')








