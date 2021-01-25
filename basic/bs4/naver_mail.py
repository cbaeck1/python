from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests


headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36"}
url = "https://mail.naver.com"
res = requests.get(url, headers=headers)
res.raise_for_status()

pprint(res.text)
with open("data/mymailnaver.html", "w", encoding="utf8") as f:
    f.write(res.text)

