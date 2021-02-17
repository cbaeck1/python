'''
클립 믹싱
연결 : concatenate_videoclips

'''

from moviepy.editor import VideoFileClip, concatenate_videoclips

# 1. 연결 : concatenate_videoclips
clip1 = VideoFileClip("basic/media/Mountains.mp4")
clip2 = VideoFileClip("basic/media/Ocean.mp4")
clip3 = VideoFileClip("basic/media/Woman_Mask.mp4")
final_clip = concatenate_videoclips([clip1,clip2,clip3])
final_clip.write_videofile("my_concatenation.mp4")
