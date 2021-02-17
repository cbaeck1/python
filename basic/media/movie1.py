'''
pip install moviepy
pip install ez_setup

MoviePy 는 소프트웨어 ffmpeg 를 사용하여 비디오 및 오디오 파일을 읽고 내 보냅니다. 
또한 (선택적으로) ImageMagick 을 사용하여 텍스트를 생성하고 GIF 파일을 작성합니다. 
다른 미디어의 처리는 Python 의 빠른 숫자 라이브러리 Numpy 에 의해 보장됩니다. 
고급 효과 및 향상은 Python 의 수많은 이미지 처리 라이브러리 (PIL, Scikit-image, scipy 등) 중 일부를 사용합니다.
MoviePy 란 python 을 이용한 Video Processing 에 최적화된 library 입니다. 
한 줄 코드로 직관적이면서도 빠르게 비디오 및 오디오의 합성이나 애니메이션, GIF 파일 생성 등을 할 수 있습니다.

어떤 경우에 써야 하나요?
    여러 비디오에 대해 처리할 때
    여러 비디오를 복잡한 방식으로 합칠 때
    video effect 를 추가하고 싶을 때(다른 video editor 없이)
    여러 이미지를 이용해 GIF 를 만들고 싶을 때
어떤 경우에 쓰면 안되나요?
    frame-by-frame 의 비디오 분석에 사용할 때 -> OpenCV 와 같은 좋은 library 가 있음
    단순히 비디오 파일을 이미지로 쪼개고 싶을 때 -> OpenCV 와 같은 좋은 library 가 존재

MoviePy 특징
    간단하며 직관적임
    Flexible 함
    Protable 함
    numpy 와의 호환성
단점
    stream video 에 대한 작업엔 적합하지 않습니다. 
    비디오의 개수가 많을 경우(100개 이상) 적합하지 않습니다.

주요 기능


'''


# 동영상을로드하고 볼륨을 낮추고 처음 5 초 동안 동영상 중앙에 제목을 추가 한 다음 결과를 파일에 기록

from moviepy.editor import *
videoclip = VideoFileClip("basic/media/Mountains.mp4")

# Load Mountains.mp4 and select the subclip 00:00:00 - 00:00:05
clip = VideoFileClip("basic/media/Mountains.mp4").subclip(0,5)

# Reduce the audio volume (volume x 0.8)
clip = clip.volumex(0.8)

# Generate a text clip. You can customize the font, color, etc.
txt_clip = TextClip("My Mountains 2013", fontsize=70, color='white')

# Say that you want it to appear 10s at the center of the screen
txt_clip = txt_clip.set_pos('center').set_duration(5)

# Overlay the text clip on the first video clip
video = CompositeVideoClip([clip, txt_clip])

# Write the result to a file (many options available !)
video.write_videofile("myMountains_edited.webm")



'''

audioclip = videoclip.audio
audioclip.write_audiofile("basic/media/audio.mp3")

# get frame 함수를 이용해 특정 시간대의 frame 추출
# 특정 초의 frame을 numpy array로 추출
img = videoclip.get_frame(10)
print(img)

clip1 = VideoFileClip("basic/media/Mountains.mp4")
# subclip을 이용해 정해진 시간 초 내의 frame만 불러옵니다.
clip2 = VideoFileClip("basic/media/Mountains.mp4").subclip(50,60)
clip3 = VideoFileClip("basic/media/Mountains.mp4")

# concat함수를 이용해 비디오를 합쳐줍니다.
final_clip = concatenate_videoclips([clip1,clip2,clip3])
final_clip.write_videofile("my_concatenation.mp4")

# 서로 다른 video와 audio 합성
videoclip = VideoFileClip("basic/media/Mountains.mp4").subclip(1, 10)
audioclip = AudioFileClip("audio.mp3").subclip(1, 10)

videoclip.audio = audioclip
videoclip.write_videofile("newVideo.mp4")

# Memory 관리
# close 함수를 이용해 비디오를 닫아줍니다.
videoclip.close()

# 여러 video clip을 합칠 때, overflow가 안나도록
parent_clip = VideoFileClip("basic/media/Mountains.mp4")
clip_list = []
for part in parent_clip:
    time_start = part[0]
    time_end = part[1]
    clip_list.append(
        parent_clip.subclip(time_start, time_end)
    )
concat_clip = concatenate_videoclips(clip_list)
'''
