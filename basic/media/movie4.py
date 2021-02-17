'''
클립 믹싱
CompositeVideoClip

'''

from moviepy.editor import VideoFileClip, CompositeVideoClip

# CompositeVideoClips
clip1 = VideoFileClip("basic/media/Road.mp4")  # 12 sec
clip2 = VideoFileClip("basic/media/Ocean.mp4") # 46 sec
clip3 = VideoFileClip("basic/media/Mountains.mp4").resize(0.60) # downsize 60%  # 8 sec

# clip2 on top of clip1, clip3 on top of clip1, and clip2
video = CompositeVideoClip([clip1,clip2,clip3])
video.write_videofile("basic/media/my_video.mp4")

# 최종 컴포지션의 크기를 지정
video2 = CompositeVideoClip([clip1,clip2,clip3], size=(2400,1800))
video2.write_videofile("basic/media/my_video2.mp4")

# 시작 및 중지 시간
video3 = CompositeVideoClip([clip1, # starts at t=0
                            clip2.set_start(5), # start at t=5s
                            clip3.set_start(9)]) # start at t=9s
video3.write_videofile("basic/media/my_video3.mp4")

# 1 초의 페이드 인 효과
video4 = CompositeVideoClip([clip1, # starts at t=0
                            clip2.set_start(5).crossfadein(1),
                            clip3.set_start(9).crossfadein(1.5)])
video4.write_videofile("basic/media/my_video4.mp4")

# 클립 위치 지정
video5 = CompositeVideoClip([clip1,
                           clip2.set_position((45,150)),
                           clip3.set_position((90,100))])
video5.write_videofile("basic/media/my_video5.mp4")
