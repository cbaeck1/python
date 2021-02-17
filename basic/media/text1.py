
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mpy
from moviepy.video.io.bindings import PIL_to_npimage

text_kr = ""
with open("basic/tts/input.txt", "r", encoding='utf-8') as f:
    for line in f:
        text_kr += line.replace(".", ". ")

print(text_kr)

# For speed we will use PIL/Pillow to draw the texts
# instead of the simpler/slower TextClip() option
font = ImageFont.truetype('c:/Windows/Fonts/NanumBarunGothic.ttf', size=60)
# txt_clip = TextClip(text_kr, fontsize=20, color='white')
# txt_clip = txt_clip.set_pos('center').set_duration(30)

# video = CompositeVideoClip([txt_clip, videoclip])
# video.write_videofile("basic/media/my_text.mp4")
# Image.new("RGBA", (1920, 1088), (255,255,255,0))
def makeframe(t):
    im = Image.new('RGB',(1920,1088))
    draw = ImageDraw.Draw(im)
    # draw.text((50, 25), "%.02f"%(t), font=font)
    draw.multiline_text((100, 200), text_kr, fill="white", font=font, align="left")
    return PIL_to_npimage(im)

clip = mpy.VideoClip(makeframe, duration=3)
# clip.preview()

one_hour_filename = "basic/media/my_text1.mp4" # Or .ogv, .webm
clip.write_videofile(one_hour_filename, fps=25)
