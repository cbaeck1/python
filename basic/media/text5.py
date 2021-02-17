# 텍스트를 이미지로 만들어서 훈련 데이터로 사용하기 위한 목적
from PIL import Image,ImageDraw,ImageFont
 
# 이미지로 출력할 글자 및 폰트 지정 (나눔고딕 트루타입폰트 파일)
draw_text = '가'
# font = ImageFont.truetype("./NanumGothic.ttf", 25)
font = ImageFont.truetype("./BMEuljiro10yearslater.ttf", 200)
 
# 이미지 사이즈 지정
text_width = 200
text_height = 200
 
# 이미지 객체 생성 (배경 검정)
canvas = Image.new('RGB', (text_width, text_height), "black")
 
# 가운데에 그리기 (폰트 색: 하양)
draw = ImageDraw.Draw(canvas)
w, h = font.getsize(draw_text)
draw.text(((text_width-w)/2.0,(text_height-h)/2.0), draw_text, 'white', font)
 
# 가.png로 저장 및 출력해서 보기
canvas.save(draw_text+'.png', "PNG")
canvas.show()
