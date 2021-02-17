
from PIL import Image, ImageDraw, ImageFont

# font = ImageFont.truetype('C:/Users/user/AppData/Local/Microsoft/Windows/Fonts/godoMaum.ttf', size=60)
font = ImageFont.truetype('C:/Windows/Fonts/BMEuljiro10yearslater.ttf', size=80, encoding='utf-8')

def makeImageUsingFont():
    max_text_width = 0
    max_text_height = 0
    lines = 0
    text_kr = ""
    with open("basic/tts/input.txt", "r", encoding='utf-8') as f:
        for line in f:
            lines += 1
            text_width, text_height = font.getsize(line)
            max_text_width = max(max_text_width, text_width)
            max_text_height = max(max_text_height, text_height)
            text_kr += line.replace(".", ". ")

    print(text_kr)
    # 이미지 사이즈 지정
    print(max_text_width, max_text_height)

    im = Image.new("RGBA", (int(text_width*1.4), int(text_height*lines*1.2)), (255,255,255,0))
    # im = Image.new("RGBA", (1920, 1088), (255,255,255,0))
    d = ImageDraw.Draw(im)
    d.multiline_text((50, 50), text_kr, fill="green", font=font, align="left")

    return im

makeImageUsingFont().show()



