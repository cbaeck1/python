import keras
import matplotlib.pyplot as plt
import image

# 영화 리뷰 분류: 이진 분류 예제
# 리뷰 텍스트를 기반으로 영화 리뷰를 긍정과 부정로 분류하는 법
# 인터넷 영화 데이터베이스로부터 가져온 양극단의 리뷰 50,000 개로 이루어진 IMDB 데이터셋을 사용
# 이 데이터셋은 훈련 데이터 25,000 개와 테스트 데이터 25,000 개로 나뉘어 있고 각각 50%는 부정, 50%는 긍정 리뷰로 구성

from keras.datasets import imdb
# 매개변수 num_words=10000 은 훈련 데이터에서 가장 자주 나타나는 단어 10,000 개만 사용하겠다는 의미
# 드물게 나타나는 단어는 무시
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data.shape, len(train_data), train_labels.shape, len(train_labels))
print(train_data[0]) # 
print(train_labels) # train_labels와 test_labels는 부정을 나타내는 0과 긍정을 나타내는 1의 리스트

# 가장 자주 등장하는 단어 10,000개로 제한했기 때문에 단어 인덱스는 10,000을 넘지 않습니다:
print(max([max(sequence) for sequence in train_data]))

# word_index는 단어와 정수 인덱스를 매핑한 딕셔너리입니다
idx = 125
word_index = imdb.get_word_index()
# 정수 인덱스와 단어를 매핑하도록 뒤집습니다
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 리뷰를 디코딩합니다. 
# 0, 1, 2는 '패딩', '문서 시작', '사전에 없음'을 위한 인덱스이므로 3을 뺍니다
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[idx]])
print("decoded_review:", decoded_review)

# 1 데이터 준비 
# 신경망에 숫자 리스트를 주입할 수는 없습니다. 리스트를 텐서로 바꾸는 두 가지 방법
# 1) 같은 길이가 되도록 리스트에 패딩을 추가하고 (samples, sequence_length) 크기의 정수 텐서로 변환합니다.
#    그 다음 이 정수 텐서를 다룰 수 있는 층을 신경망의 첫 번째 층으로 사용합니다(Embedding 층을 말하며 나중에 자세히 다루겠습니다).
# 2) 리스트를 원-핫 인코딩하여 0과 1의 벡터로 변환합니다. 
#    예를 들면 시퀀스 [3, 5]를 인덱스 3과 5의 위치는 1이고 그 외는 모두 0인 10,000차원의 벡터로 각각 변환합니다. 
#    그 다음 부동 소수 벡터 데이터를 다룰 수 있는 Dense 층을 신경망의 첫 번째 층으로 사용합니다.
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # 크기가 (len(sequences), dimension))이고 모든 원소가 0인 행렬을 만듭니다
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # results[i]에서 특정 인덱스의 위치를 1로 만듭니다
    return results

# 훈련 데이터를 벡터로 변환합니다
x_train = vectorize_sequences(train_data)
# 테스트 데이터를 벡터로 변환합니다
x_test = vectorize_sequences(test_data)
print(x_train.shape, len(x_train), x_test.shape, len(x_test))

# 레이블을 벡터로 바꿉니다
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(y_train.shape, len(y_train), y_test.shape, len(y_test))

# 2. 신경망 모델 만들기
from keras import models
from keras import layers

model = models.Sequential()
# 16개의 은닉 유닛을 가진 두 개의 은닉층
# 현재 리뷰의 감정을 스칼라 값의 예측으로 출력하는 세 번째 층 : sigmoid는 임의의 값을 [0, 1] 사이로
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 케라스에 rmsprop, binary_crossentropy, accuracy가 포함되어 있기 때문에 
# 옵티마이저, 손실 함수, 측정 지표를 문자열로 지정하는 것이 가능
# 옵티마이저의 매개변수를 바꾸거나 자신만의 손실 함수, 측정 함수를 전달
# 옵티마이저 파이썬 클래스를 사용해 객체를 직접 만들어 optimizer 매개변수에 전달
'''
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# loss와 metrics 매개변수에 함수 객체를 전달             
from keras import losses
from keras import metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
'''

# 3. 훈련 검증
# 원본 훈련 데이터에서 10,000의 샘플을 떼어서 검증 세트로
# Train on 15000 samples, validate on 10000 samples 
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 모델을 512개 샘플씩 미니 배치를 만들어 30번의 에포크 동안 훈련
epochsCnt = 30
train_result = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=epochsCnt,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = train_result.history
history_dict.keys()

# 
acc = train_result.history['accuracy']
val_acc = train_result.history['val_accuracy']
loss = train_result.history['loss']
val_loss = train_result.history['val_loss']
epochs = range(1, len(acc) + 1)

# ‘bo’는 파란색 점을 의미합니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# ‘b’는 파란색 실선을 의미합니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
image.save_fig("3.4.1 Training and validation loss")  
plt.show()


plt.clf()   # 그래프를 초기화합니다
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
image.save_fig("3.4.2 Training and validation accuracy")  
plt.show()


plt.clf()   # 그래프를 초기화합니다
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, acc, 'ro', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation loss & accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss & Accuracy')
plt.legend()
image.save_fig("3.4.3 Training and validation loss & accuracy")  
plt.show()



# 처음부터 다시 새로운 신경망을 4번의 에포크 동안만 훈련하고 테스트 데이터에서 평가
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print("results", results)

# 훈련된 모델로 새로운 데이터에 대해 예측하기
pred = model.predict(x_test)
print("pred", pred)


# 정리
'''
원본 데이터를 신경망에 텐서로 주입하기 위해서는 꽤 많은 전처리가 필요합니다. 단어 시퀀스는 이진 벡터로 인코딩될 수 있고 다른 인코딩 방식도 있습니다.
relu 활성화 함수와 함께 Dense 층을 쌓은 네트워크는 (감성 분류를 포함하여) 여러 종류의 문제에 적용할 수 있어서 앞으로 자주 사용하게 될 것입니다.
(출력 클래스가 두 개인) 이진 분류 문제에서 네트워크는 하나의 유닛과 sigmoid 활성화 함수를 가진 Dense 층으로 끝나야 합니다. 이 신경망의 출력은 확률을 나타내는 0과 1 사이의 스칼라 값입니다.
이진 분류 문제에서 이런 스칼라 시그모이드 출력에 대해 사용할 손실 함수는 binary_crossentropy입니다.
rmsprop 옵티마이저는 문제에 상관없이 일반적으로 충분히 좋은 선택입니다. 걱정할 거리가 하나 줄은 셈입니다.
훈련 데이터에 대해 성능이 향상됨에 따라 신경망은 과대적합되기 시작하고 이전에 본적 없는 데이터에서는 결과가 점점 나빠지게 됩니다. 항상 훈련 세트 이외의 데이터에서 성능을 모니터링해야 합니다.
'''


