'''
webdriver와 actionchains를 사용하여 구글 지메일을 자동으로 보내는 프로그램을 만드는 예제


'''
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

import time

driver = webdriver.Chrome()
url = 'https://google.com'
driver.get(url)
#driver.maximize_window()
action = ActionChains(driver)

# selector #gb > div > div.gb_Ue > a
# xpath : //*[@id="gb"]/div/div[2]/a
# /html/body/div[1]/div[1]/div/div/div/div[2]/a
# driver.find_element_by_xpath('//*[@id="gb"]/div/div[2]/a').click()
driver.find_element_by_css_selector('#gb > div > div.gb_Ue > a').click()

action.send_keys('cbaeck1').perform()
action.reset_actions()
# #identifierNext > div > button > div.VfPpkd-Jh9lGc
# //*[@id="identifierNext"]/div/button/div[1]
driver.find_element_by_css_selector('#identifierNext > div > button > div.VfPpkd-Jh9lGc').click()

time.sleep(2)
driver.find_element_by_css_selector('.whsOnd.zHQkBf').send_keys('본인비밀번호')
driver.find_element_by_css_selector('.CwaK9').click()
time.sleep(2)

driver.get('https://mail.google.com/mail/u/0/?ogbl#inbox')
time.sleep(2)

driver.find_element_by_css_selector('.T-I.J-J5-Ji.T-I-KE.L3').click()
time.sleep(1)

send_buton = driver.find_element_by_css_selector('.gU.Up')

(
action.send_keys('보낼메일주소').key_down(Keys.ENTER).pause(2).key_down(Keys.TAB)
.send_keys('제목입니다.').pause(2).key_down(Keys.TAB)
.send_keys('abcde').pause(2).key_down(Keys.ENTER)
.key_down(Keys.SHIFT).send_keys('abcde').key_up(Keys.SHIFT).pause(2)
.move_to_element(send_buton).click()
.perform()
)