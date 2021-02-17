'''
클립 속성을 변경하는 방법

'''

from moviepy.editor import *

clip = (VideoFileClip("basic/media/Mountains.mp4")
           .fx( vfx.resize, width=460) # resize (keep aspect ratio)
           .fx( vfx.speedx, 2) # double the speed
           .fx( vfx.colorx, 0.5)) # darken the picture

clip.write_videofile("basic/media/my_clip.mp4")

