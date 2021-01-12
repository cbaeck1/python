# 데이터 로드와 사전처리 데이터 : CSV 데이터로드
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

# 파일에서 tf.data.Dataset 로 CSV 데이터를로드하는 방법의 예
# 이 튜토리얼에서 사용 된 데이터는 타이타닉 승객 목록에서 가져온 것
# 이 모델은 연령, 성별, 티켓 등급, 혼자 여행하는지 여부와 같은 특성을 기반으로 승객이 생존 할 가능성을 예측






