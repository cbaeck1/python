# reply_crawling/reply_crawling_def.py
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

def get_replys(url=None,chromedriver_path=None,excel_name=None,keyword=None,imp_time=1):
    if not url:
        print("url은 필수 인자 입니다.")
        return

    if not excel_name and not keyword:
        print("excel_name, keyword 둘 중 하나는 반드시 필요합니다.")
        return

    ## 댓글 페이지가 아닌 그냥 뉴스 페이지라면
    if not 'm_view=1' in url :
        url += '&m_view=1' # 인자를 추가하여 댓글 페이지로 이동하게 만든다.

    ## 창열기 + 접속
    print("[접속]")
    driver = webdriver.Chrome(chromedriver_path)
    driver.implicitly_wait(imp_time) # 값이 클수록 요소가 없다고 판단하는 대기시간이 길어진다.
    driver.maximize_window()
    driver.get(url)

    ## 더보기 버튼 누르기
    print("[더보기 클릭 중]")
    attempt = 0 # 더보기 횟수
    while True:
        try:
            driver.find_element_by_css_selector('span.u_cbox_page_more').click()
            attempt = 0 # 더보기를 눌렀으니 다시 카운트
        except:
            attempt += 1
            if attempt > 5 :
                break # 더이상 더보기 버튼이 없다고 판단.

    ## 댓글 요소 찾기
    print("[댓글 요소 찾기]")
    replys = driver.find_elements_by_css_selector('ul.u_cbox_list>li.u_cbox_comment')
    # print(len(replys))

    ## 댓글 내용 수집
    if excel_name or keyword:
        print("[댓글 내용 수집]")
        results = []
        specific_results = []
        for index, reply in enumerate(replys):
            try:
                author = reply.find_element_by_css_selector('span.u_cbox_nick').text
                content = reply.find_element_by_css_selector('span.u_cbox_contents').text
                results.append((author,content))
                ## 특정 단어 있는 경우는 따로 또 저장
                if keyword and keyword in content:
                    specific_results.append((index, author, content))
            except:
                pass # 삭제된 댓글 스킵

    ## 상단 메뉴 바를 숨기기
    print("[상단 메뉴바 숨기기]")
    header = driver.find_element_by_css_selector('#header') # 메뉴 바 요소 찾기
    driver.execute_script("arguments[0].style.display='none'", header) # execute_script : 자바스크립트 문법 실행
    # 또는 driver.execute_script("document.querySelector('#header').style.display='none'")

    ## 폴더 생성
    if keyword:
        print("[폴더 생성]")
        import os
        folder_name = keyword
        if not os.path.isdir('./{}'.format(folder_name)):
            os.mkdir('./{}'.format(folder_name))

        ## 캡처하기 - 2
        print("[캡처 시작]")
        from PIL import Image
        from io import BytesIO

        for index, s in enumerate(specific_results):
            # 해당 요소까지 스크롤
            driver.execute_script('arguments[0].scrollIntoView(true);', replys[s[0]])

            # 현재 화면 캡처하기
            img = driver.get_screenshot_as_png() #이진 형태로 저장

            ## 요소 좌표를 추출하고 '현재화면캡처사진'에서 잘라낸 뒤 저장하기
            location = replys[s[0]].location_once_scrolled_into_view # 현재 화면에서 해당 요소가 있는는지 dict로 반환
            size = replys[s[0]].size # 해당 요소의 크기를 dict로 반환

            left = location['x']
            top = location['y']
            right = location['x'] + size['width'] + 20
            bottom = location['y'] + size['height'] + 20
            box = (left,top,right,bottom)

            if location:
                im = Image.open(BytesIO(img)) # Image.open()에 이진형태를 바로 넣으려면 BytesIO가 필요
                im = im.crop(box)
                im.save('./{0}/{0}{1}.png'.format(folder_name,index))

    ## 엑셀로 저장
    if excel_name:
        print(["전체 댓글 엑셀로 저장"])
        import pandas as pd #pandas, openpyxl
        col =[ '작성자','내용']
        data_frame = pd.DataFrame(results,columns=col)
        data_frame.to_excel('{}.xlsx'.format(excel_name),sheet_name='{}'.format(excel_name),startrow=0,header=True)

    ## 닫기
    driver.quit()


## 메인 영역
if __name__ == '__main__':
    url = "https://news.naver.com/main/read.nhn?sid1=001&oid=001&aid=0011335704" # 댓글 페이지가 아닌 뉴스 페이지로
    excel_name = "댓글 수집"
    keyword = "윈도우"
    chrome = "chromedriver.exe"
    get_replys(url,chrome,excel_name,keyword)

