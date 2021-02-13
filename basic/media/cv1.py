'''
pip install opencv-python

OpenCV (Open source Computer Vision)는 실시간 컴퓨터 비젼을 처리하는 목적으로 만들어진 라이브러리로서, 
인텔에 의해 처음 만들어져서 현재는 Itseez (2016년 인텔에 편입)에 의해 지원되고 있다. 
OpenCV는 크로스 플랫폼 라이브러리로서 윈도우즈, 리눅스, Max 등에서 무료로 사용할 수 있다. 
OpenCV는 기본적으로 C++로 쓰여져 있는데, 이 라이브러리는 C/C++, Python, Java, C#, Ruby 등 여러 언어에서 사용할 수 있다

1. 이미지 파일 읽고 쓰기
2. 카메라 영상 처리
3. 카메라 영상 저장하기
4. OpenCV와 Matplotlib 활용



'''

import cv2
 
# 이미지 읽기
img = cv2.imread('basic/python.jpg', 1)
 
# 이미지 화면에 표시
cv2.imshow('Test Image', img)
cv2.waitKey(0)
# 이미지 윈도우 삭제
cv2.destroyAllWindows()
 
# 이미지 다른 파일로 저장
cv2.imwrite('basic/media/test2.png', img)

# 동영상 파일에서 읽기
# cap = cv2.VideoCapture(0)   # 0: default camera
# #cap = cv2.VideoCapture("test.mp4") 
 
# while cap.isOpened():
#     # 카메라 프레임 읽기
#     success, frame = cap.read()
#     if success:
#         # 프레임 출력
#         cv2.imshow('Camera Window', frame)
 
#         # ESC를 누르면 종료
#         key = cv2.waitKey(1) & 0xFF
#         if (key == 27): 
#             break
 
# cap.release()
# cv2.destroyAllWindows()

# 카메라 영상 저장하기
# cap = cv2.VideoCapture(0); 
 
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print("size: {0} x {1}".format(width, height))
 
# # 영상 저장을 위한 VideoWriter 인스턴스 생성
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# writer = cv2.VideoWriter('test.avi', fourcc, 24, (int(width), int(height)))
 
# while cap.isOpened():
#     success, frame = cap.read()
#     if success:
#         writer.write(frame)  # 프레임 저장
#         cv2.imshow('Video Window', frame)
 
#         # q 를 누르면 종료
#         if cv2.waitKey(1) & 0xFF == ord('q'): 
#             break
#     else:
#         break
 
# cap.release()
# writer.release()  # 저장 종료
# cv2.destroyAllWindows()

# 
from matplotlib import pyplot as plt
 
img = cv2.imread('basic/happyfish.png', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

