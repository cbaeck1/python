# 데이터 로드와 사전처리 데이터 : pandas.DataFrame로드
# 목차
# 1. 데이터로드
# 2. 데이터전처리
#   2.1 연속데이터
#   2.2 범주형데이터
#   2.3 결합된 전처리 레이어
# 3. 모델 구축
# 4. 훈련, 평가 및 예측


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
