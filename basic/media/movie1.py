'''
어떤 경우에 써야 하나요?
    여러 비디오에 대해 처리할 때
    여러 비디오를 복잡한 방식으로 합칠 때
    video effect를 추가하고 싶을 때(다른 video editor 없이)
    여러 이미지를 이용해 GIF를 만들고 싶을 때
어떤 경우에 쓰면 안되나요?
    frame-by-frame의 비디오 분석에 사용할 때 -> OpenCV와 같은 좋은 library가 있음
    단순히 비디오 파일을 이미지로 쪼개고 싶을 때 -> OpenCV와 같은 좋은 library가 존재


주요 기능
기본 단위는 clips로 불리며, 크게 AudioClips, VideoClips, AudioFileClip, VideoFileClip 클래스로 구성




'''

from movipy.editor import *
videoclip = VideoFileClip("myvideo.mp4")
audioclip = videoclip.audio

# get frame 함수를 이용해 특정 시간대의 frame 추출
img = vid_clip.get_frame(10)

from moviepy.editor import VideoFileClip, concatenate_videoclips
clip1 = VideoFileClip("myvideo.mp4")
# subclip을 이용해 정해진 시간 초 내의 frame만 불러옵니다.
clip2 = VideoFileClip("myvideo2.mp4").subclip(50,60)
clip3 = VideoFileClip("myvideo3.mp4")
# concat함수를 이용해 비디오를 합쳐줍니다.
final_clip = concatenate_videoclips([clip1,clip2,clip3])
final_clip.write_videofile("my_concatenation.mp4")
