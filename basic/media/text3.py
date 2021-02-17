
from moviepy.editor import *

videoclip = VideoFileClip("basic/media/Cross.mp4")

txt = "\n".join([
"A long time ago, in a faraway galaxy,",
"there lived a prince and a princess",
"who had never seen the stars, for they",
"lived deep underground.",
"",
"Many years before, the prince's",
"grandfather had ventured out to the",
"surface and had been burnt to ashes by",
"solar winds.",
"",
"One day, as the princess was coding",
"and the prince was shopping online, a",
"meteor landed just a few megameters",
"from the couple's flat."
])

txt_clip = TextClip(txt, fontsize=70, color='white')
txt_clip = txt_clip.set_pos('center').set_duration(30)

video = CompositeVideoClip([videoclip, txt_clip])
video.write_videofile("basic/media/my_text3.mp4")

