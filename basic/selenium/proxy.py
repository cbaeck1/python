'''
무료 프록시 찾기 
구글에 free proxy 등으로 검색하면 여러 사이트가 나옵니다. 그중 적당한 사이트에 들어가서 선택하면 됩니다. 
IP Address와 Port를 복사하여 사용하면 됩니다

프록시 설정 
PROXY = "IP:Port"
webdriver.DesiredCapabilities.CHROME['proxy'] = {
    "httpProxy": PROXY,
    "ftpProxy": PROXY,
    "sslProxy": PROXY,
    "proxyType": "MANUAL"
}

driver = webdriver.Chrome()
driver.get("URL")
'''

from selenium import webdriver

PROXY = "117.1.16.131:8080" # IP:Port

webdriver.DesiredCapabilities.CHROME['proxy'] = {
    "httpProxy": PROXY,
    "ftpProxy": PROXY,
    "sslProxy": PROXY,
    "proxyType": "MANUAL"
}

driver = webdriver.Chrome()
driver.get("https://www.google.com")


