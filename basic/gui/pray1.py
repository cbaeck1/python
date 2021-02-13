import pyautogui
import openpyxl
import pyperclip

print('Press Ctrl-C to quit.')
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.25
shift = 0
START_POSITION = 1001
MAXCNT = 30

# 엑셀파일 열기
wb = openpyxl.load_workbook('basic/gui/pray.xlsx')
# 현재 Active Sheet 얻기
ws = wb.active
row_index = START_POSITION + 1

try:
    while True:
        # 376 518 650 950
        pyautogui.moveTo(376, 518, 1)
        pyautogui.click()
        pyautogui.dragTo(900, 1200 - shift*20, 1, button='left')

        # ctrl + c 키를 입력합니다. 
        pyautogui.hotkey('ctrl', 'c') 
        tmp = pyperclip.paste()
        print(tmp)
        if tmp.find('아멘') == -1 and shift < MAXCNT:
            shift += 1
            continue
        elif shift != 0:
            shift = 0        

        print(row_index)
        ws.cell(row=row_index, column=1).value = tmp
        if row_index >= 1000 + START_POSITION:
            break
        row_index += 1

        # 3905 197
        pyautogui.moveTo(905, 197, 1)
        pyautogui.click()

    # 엑셀 파일 저장
    wb.save('basic/gui/pray2.xlsx')
    wb.close()


except KeyboardInterrupt:
    print('\n')