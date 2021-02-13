'''
pip install pyautogui
https://pyautogui.readthedocs.io/en/latest/keyboard.html




'''
import pyautogui

# 좌표 객체 얻기 
position = pyautogui.position()

# 화면 전체 크기 확인하기
print(pyautogui.size())

# x, y 좌표
print(position.x, position.y)

# 마우스 이동 (x 좌표, y 좌표)
pyautogui.moveTo(500, 500)

# 마우스 이동 (x 좌표, y 좌표 2초간)
pyautogui.moveTo(100, 100, 2)  

# 마우스 이동 ( 현재위치에서 )
print('이동')
pyautogui.moveRel(200, 300, 2)

# 마우스 클릭
print('클릭')
pyautogui.click()

# 2초 간격으로 2번 클릭
print('2번 클릭')
pyautogui.click(clicks= 2, interval=2)

# 더블 클릭
pyautogui.doubleClick()

# 오른쪽 클릭
pyautogui.click(button='right')

# 스크롤하기 
print('스크롤하기')
pyautogui.scroll(100)

# 드래그하기
# drag the mouse left 300 pixels over 2 seconds while holding down the right mouse button
print('드래그하기')
pyautogui.drag(300, 0, 2, button='left')

# drag mouse to X of 1000, Y of 200 over 2 seconds while holding down left mouse button
pyautogui.dragTo(1000, 800, 2, button='left')

pyautogui.moveTo(100, 100, 2, pyautogui.easeInQuad)     # start slow, end fast
pyautogui.moveTo(200, 200, 2, pyautogui.easeOutQuad)    # start fast, end slow

