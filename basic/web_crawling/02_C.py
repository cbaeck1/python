'''
C. 1 to 50 봇

1. 1to50 직접해보기
    http://zzzscore.com/1to50/
2. 1to50 코드 분석
크롬 개발자 도구를 이용하면 알겠지만, 버튼처럼 보일 뿐 사실 영역을 잡아주는 div 태그를 사용하고 있습니다. 
해야할 작업은 크게 3가지
1. 게임에 사용되는 모든 버튼 요소 정보를 가져온다.
2. 각 버튼(영역)의 내부내용(.text)를 파악한다. 
3. 우리가 찾는 숫자면 클릭한다.

먼저 5x5에서 (1,1)위치에 있는 버튼의 xpath를 copy하여 확인해보고, (1,2), (1,3) 도 확인하면 다음과 같습니다.
(1,1)위치 div의 xpath : //*[@id="grid"]/div[1]
(1,2)위치 div의 xpath : //*[@id="grid"]/div[2]
(1,3)위치 div의 xpath : //*[@id="grid"]/div[3]
..
(5,5)위치 div의 xpath : //*[@id="grid"]/div[25]
즉, //*[@id="grid"]/div[xx] 형태로 되있는 것을 모두 감지하기위해, 변하는 부분에 *를 넣어 //*[@id="grid"]/div[*] 로 해당 요소를 전부 탐색해봅시다.

3. 1번 누르기
가져온 버튼 요소 중 텍스트가 1인 것을 찾아 클릭하도록 코드를 작성해봅시다.
현재 검색된 버튼들을 모두 검사하다가, 만일 1인 버튼을 찾으면 클릭하도록 반복문 for를 작성합니다.

4. 1~50까지 클릭

'''

from selenium import webdriver

driver = webdriver.Chrome('chromedriver')
driver.get('http://zzzscore.com/1to50')
driver.implicitly_wait(3)

btns = driver.find_elements_by_xpath('//*[@id="grid"]/div[*]')
print(len(btns))
print(btns[0].text) #0번 요소의 텍스트
print()

# 3. 1번 누르기
# for btn in btns:
#    # print(btn.text)
#    if btn.text == "1":
#        btn.click()

# 
# 전역변수 : 현재 찾아야될 숫자
num = 1
def clickBtn():
    global num
    btns = driver.find_elements_by_xpath('//*[@id="grid"]/div[*]')

    for btn in btns:
        # print(btn.text, end='\t')
        if btn.text == str(num):
            btn.click()
            print(num)
            num += 1
            return

while num <= 50:
    clickBtn()
