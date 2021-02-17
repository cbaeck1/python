from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype("c:/Windows/Fonts/NanumBarunGothic.ttf", 48)
im = Image.new("RGB", (200, 200), "white")
d = ImageDraw.Draw(im)
d.line(((0, 100), (200, 100)), "gray")
im.show()

d.line(((100, 0), (100, 200)), "gray")
im.show()

# 특정 앵커 를 사용하여 텍스트를 그릴 때 지정된 앵커 포인트가 xy좌표에 있도록 텍스트가 배치
# l:left, m:middle, r:right
# a:ascender, t:top, m:middle, s:baseline, b:bottom, d:descender
d.text((100, 100), "Quick", fill="black", anchor="ms", font=font, align="center")

im.show()

# def makeframe(t):
#     im = plim.new('RGB',(1024,768))
#     draw = ImageDraw.Draw(im)
#     # draw.text((50, 25), "%.02f"%(t), font=font)
#     draw.multiline_text((100, 200), text_kr, fill="white", font=font, align="center")
#     return PIL_to_npimage(im)

# clip = mpy.VideoClip(makeframe, duration=3)
# clip.preview()
