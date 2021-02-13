'''
이미지로 좌표찾기

계산기에서 5번 버튼을 누른다고 가정
5번 버튼 이미지의 스크린 샷을 5.png로 저장

'''

import pyautogui as pg

# screenshot
# 전체
# im1 = pg.screenshot()
# im2 = pg.screenshot('basic/gui/my.png')

# 캡처 할 영역의 왼쪽, 위쪽, 너비 및 높이의 4 개 정수 튜플
# 80 916 208 1011
# im = pg.screenshot('basic/gui/7.png', region=(80, 916, 208-80, 1010-916))

# 계산기에서 5번 버튼을 누른다고 가정
# 이미지가 있는 위치를 가져옵니다. 
button7location = pg.locateOnScreen('basic/gui/7.png') 
print(button7location)

# Box 객체의 중앙 좌표를 리턴합니다. 
point7 = pg.center(button7location) 
print(point7)

# clicks the center of where the 7 button was found
pg.click(point7.x, point7.y)
# a shortcut version to click on the center of where the 7 button was found
# pg.click('basic/gui/7.png')

# 선택적 confidence키워드 인수는 함수가 화면에서 이미지를 찾는 정확도를 지정합니다. 
# 이는 무시할 수있는 픽셀 차이로 인해 함수가 이미지를 찾을 수없는 경우에 유용합니다.
button7 = pg.locateOnScreen('basic/gui/7.png', confidence=0.9) 
print(button7)

# x, y = pg.locateCenterOnScreen('basic/gui/7.png')
# x, y = pg.center(pg.locateOnScreen('basic/gui/7.png'))
# print(x, y)
# pg.click(x, y)

# region 인수 ((left, top, width, height)의 4 개 정수 튜플)를 전달하여 전체 화면 대신 화면의 더 작은 영역 만 검색
button7 = pg.locateOnScreen('basic/gui/7.png', region=(0,0, 300, 1000)) 
print(button7)

# 그레이 스케일 매칭
button7 = pg.locateOnScreen('basic/gui/7.png', grayscale=True) 
print(button7)

# 픽셀 매칭
# 스크린 샷에서 픽셀의 RGB 색상을 얻으려면 Image 객체의 getpixel()메서드를 사용
# im = pyautogui.screenshot()
# 또는 단일 함수로 pixel()이전 호출의 래퍼 인 PyAutoGUI 함수를 호출
# pix = pyautogui.pixel(100, 200)
# 단일 픽셀이 주어진 픽셀과 일치하는지 확인하기 만하면되는 경우 
# pixelMatchesColor()함수를 호출하여 나타내는 색상의 X 좌표, Y 좌표 및 RGB 튜플을 전달
# pyautogui.pixelMatchesColor(100, 200, (130, 135, 144))
# 선택적 tolerance키워드 인수는 일치하는 동안 각 빨강, 녹색 및 파랑 값이 얼마나 달라질 수 있는지 지정
# pyautogui.pixelMatchesColor(100, 200, (140, 125, 134), tolerance=10)





