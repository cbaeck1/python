# 첫 번째 신경망 훈련하기: 저장과 로드
# 목차
# 1. 저장방식
# 2. 설정
#   2.1 설치와 임포트
#   2.2 예제 데이터셋 받기
#   2.3 모델 정의
# 3. 훈련하는 동안 체크포인 저장하기
#   3.1 체크포인트 콜백 사용하기
#   3.2 체크포인트 콜백 매개변수
# 4. 이 파일들은 무엇인가요?
# 5. 수동으로 가중치 저장하기
# 6. 전체 모델 저장하기
#   6.1 SaveModel 포맷
#   6.2 HDF5 파일로 저장하기
#   6.3 사용자 정의 객체


# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("허브 버전: ", hub.__version__)
print("GPU", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
