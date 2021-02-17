'''
클립 믹싱
스태킹 : clip_array

'''

from moviepy.editor import VideoFileClip, clips_array, vfx

# 1. 연결 : clip_array
clip1 = VideoFileClip("basic/media/Mountains.mp4").margin(10) # add 10px contour
clip2 = clip1.fx( vfx.mirror_x)
clip3 = clip1.fx( vfx.mirror_y)
clip4 = clip1.resize(0.60) # downsize 60%
final_clip = clips_array([[clip1, clip2],
                          [clip3, clip4]])
final_clip.resize(width=480).write_videofile("basic/media/my_stack.mp4")

