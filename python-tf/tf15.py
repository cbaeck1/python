# 첫 번째 신경망 훈련하기: 과대적합과 과소적합
# 목차
# 1. IMDB 데이터셋 
# 2. 과대적합 예제
#   2.1 기준 모델 만들기
#   2.2 작은 모델 만들기
#   2.3 큰 모델 만들기 
#   2.4 훈련 손실과 검정 손실 그래프 그리기 
# 3. 과대적합을 방지하기 위한 전략
#   3.1 가중치를 규제하기
#   3.1 드롭아웃 추가하기

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
