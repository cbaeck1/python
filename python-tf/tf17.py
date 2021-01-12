# 첫 번째 신경망 훈련하기: Keras Tuner 소개
# 목차
# 1. 개요
# 2. 설정
# 3. 데이터 세트 다운로드 및 준비
# 4. 모델정의
# 5. 튜너를 인스턴스화하고 하이퍼 튜닝을 수행
# 6. 요약


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
