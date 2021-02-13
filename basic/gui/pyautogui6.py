import pyautogui, sys
print('찾기 시작')

# 이미지가 있는 위치를 가져옵니다. 
banana = pyautogui.locateOnScreen('basic/banana.png', confidence=0.8) 
print(banana)

# Box 객체의 중앙 좌표를 리턴합니다. 
point = pyautogui.center(banana) 
print(point)


