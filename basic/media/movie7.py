'''
사용자 지정 효과를 만드는 방법

'''

from moviepy.editor import *
import numpy as np

clip = VideoFileClip("basic/media/Mountains.mp4")

# 3배 빠르게, 상영시간은 그래도, 3초 부분만
modifiedClip1 = clip.fl_time(lambda t: 3*t, keep_duration=True).subclip(0,3)
modifiedClip1.write_videofile("basic/media/my_clip1.mp4")

# t = 0s 및 t = 2s 사이에서 진동 
modifiedClip2 = clip.fl_time(lambda t: 1+np.sin(t), keep_duration=True)
modifiedClip1.write_videofile("basic/media/my_clip2.mp4")

# 프레임의 녹색 및 파란색 채널을 반전
def invert_green_blue(image):
    return image[:,:,[0,2,1]]

modifiedClip3 = clip.fl_image(invert_green_blue)
modifiedClip3.write_videofile("basic/media/my_clip3.mp4")

# 시간과 프레임 사진을 모두 고려하여 클립을 처리
# 360 픽셀의 일정한 높이로 클립이 아래로 스크롤
def scroll(get_frame, t):
    """
    This function returns a 'region' of the current frame.
    The position of this region depends on the time.
    """
    frame = get_frame(t)
    frame_region = frame[int(t):int(t)+360,:]
    return frame_region

modifiedClip4 = clip.fl(scroll)
modifiedClip4.write_videofile("basic/media/my_clip4.mp4")



