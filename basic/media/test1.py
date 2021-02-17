from moviepy.editor import *
import numpy as np

# clip = VideoFileClip("basic/media/Mountains.mp4")

# 3배 빠르게
# modifiedClip1 = clip.fl_time(lambda t: 3*t, keep_duration=False)
# print(clip.duration/3)
# modifiedClip1.duration = clip.duration / 3
# modifiedClip1.write_videofile("basic/media/my_clip1.mp4")

# 3배 빠르게
# modifiedClip1 = clip.fl_time(lambda t: 3*t, keep_duration=True).subclip(0,3)
# # modifiedClip1.write_videofile("basic/media/my_clip1.mp4")
# modifiedClip1.show(2.5, interactive = True)

# modifiedClip1.preview() # preview with default fps=15


# clip = (VideoFileClip("basic/media/Mountains.mp4")
#            .fx( vfx.speedx, 2)) # double the speed 
# clip.write_videofile("basic/media/my_clip.mp4")


# txt = "\n".join([
# "A long time ago, in a faraway galaxy,",
# "there lived a prince and a princess",
# "who had never seen the stars, for they",
# "lived deep underground.",
# "",
# "Many years before, the prince's",
# "grandfather had ventured out to the",
# "surface and had been burnt to ashes by",
# "solar winds.",
# "",
# "One day, as the princess was coding",
# "and the prince was shopping online, a",
# "meteor landed just a few megameters",
# "from the couple's flat."
# ])

# print(txt)
import PIL.Image as plim
from PIL import ImageFont, ImageDraw
import moviepy.editor as mpy
from moviepy.video.io.bindings import PIL_to_npimage


# For speed we will use PIL/Pillow to draw the texts
# instead of the simpler/slower TextClip() option

font = ImageFont.truetype('c:/Windows/Fonts/NanumBarunGothic.ttf', size=30)
# print(os.path.basename(font))
print(str(font))
# videoclip = VideoFileClip("basic/media/Cross.mp4")
# txt_clip = TextClip("Bible 태초에 하나님이 천지를 창조하시니라.",font=str(font), fontsize=70, color="green") 
# txt_clip = txt_clip.set_pos('center').set_duration(videoclip.duration)
# txt_clip.preview()
text = 'Bible 태초에 하나님이 \n 천지를 창조하시니라'
text_kr = ""
with open("basic/tts/input.txt", "r", encoding='utf-8') as f:
    for line in f:
        text_kr += line.replace(".", ". ")

print(text_kr)

def makeframe(t):
    im = plim.new('RGB',(1024,768))
    draw = ImageDraw.Draw(im)
    # draw.text((50, 25), "%.02f"%(t), font=font)
    draw.multiline_text((100, 200), text_kr, fill="white", font=font, align="center")
    return PIL_to_npimage(im)

clip = mpy.VideoClip(makeframe, duration=3)
clip.preview()

# Write the 1h-long clip to a file (takes 2 minutes)
# You can change the extension to test other formats
one_hour_filename = "basic/media/one_hour.mp4" # Or .ogv, .webm
clip.write_videofile(one_hour_filename, fps=25)

# We now read the file produced and extract frames at several
# times. Check that the frame content matches the time.
# new_clip = mpy.VideoFileClip(one_hour_filename)
# for t in [0,1200.5, 850.2, 2000.3, 150.25, 150.25]:
#     new_clip.save_frame('%d.jpeg'%(int(100*t)),t)